"""Input-space reparameterization and elementwise coefficient protection."""

import torch
from torch import nn
from torch.nn import functional as F


LAYER_PATHS = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_o": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
}
DEFAULT_LAYER_TYPES = "attn_q,attn_k,attn_v,ffn_up,ffn_down"


def select_target_modules(model, layer_types, target_layers):
    """Return named linear projections in a model with model.layers blocks."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a decoder with model.layers (e.g. Llama).")
    count = len(model.model.layers)
    kinds = [part.strip() for part in layer_types.split(",")]
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("layer_types must contain distinct projection names.")
    unknown = set(kinds) - set(LAYER_PATHS)
    if unknown:
        raise ValueError(f"Unknown layer types: {sorted(unknown)}")
    if target_layers == "all":
        indices = list(range(count))
    else:
        indices = []
        for part in target_layers.split(","):
            bounds = part.strip().split("-")
            if len(bounds) == 1:
                indices.append(int(bounds[0]))
            elif len(bounds) == 2:
                start, end = map(int, bounds)
                if start > end:
                    raise ValueError(f"Reversed layer range: {part}")
                indices.extend(range(start, end + 1))
            else:
                raise ValueError(f"Invalid layer range: {part}")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Select at least one layer, without duplicate indices.")
    if min(indices) < 0 or max(indices) >= count:
        raise ValueError(f"Layer indices must be between 0 and {count - 1}.")
    targets = {}
    for index in sorted(indices):
        for kind in kinds:
            name = f"model.layers.{index}.{LAYER_PATHS[kind]}"
            try:
                module = model.get_submodule(name)
            except AttributeError as exc:
                raise ValueError(f"The model has no projection {name}.") from exc
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{name} must be an unquantized nn.Linear.")
            targets[name] = module
    return targets


class WSRLinear(nn.Module):
    """Represent W[out, in] as C @ U.T, with C = W @ U and fixed U[in, in].

    True mask entries are protected. The bias of a reparameterized projection
    is kept fixed. Other model parameters are controlled by the phase runner.
    """

    def __init__(self, linear, basis, mask=None):
        super().__init__()
        if basis.shape != (linear.in_features, linear.in_features):
            raise ValueError("The input basis must be square and complete.")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        basis = basis.to(device=linear.weight.device, dtype=linear.weight.dtype)
        if not torch.isfinite(basis).all():
            raise ValueError("The basis contains non-finite values.")
        self.register_buffer("basis", basis)
        with torch.no_grad():
            self.basis_coeff = nn.Parameter(linear.weight @ basis)
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().clone()
        )
        if mask is None:
            mask = torch.zeros_like(self.basis_coeff, dtype=torch.bool)
        if mask.shape != self.basis_coeff.shape:
            raise ValueError("The mask must have the same shape as the weight.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("The coefficient mask must be binary.")
        self.register_buffer("coeff_mask", mask.to(basis.device, dtype=torch.bool))
        self.has_frozen = bool(self.coeff_mask.any())
        self.register_buffer(
            "frozen_values", self.basis_coeff.detach()[self.coeff_mask].clone()
        )

    def forward(self, inputs):
        coeff = self.basis_coeff
        if self.has_frozen:
            coeff = torch.where(self.coeff_mask, coeff.detach(), coeff)
        return F.linear(inputs, coeff @ self.basis.T, self.bias)

    @torch.no_grad()
    def restore_frozen(self):
        """Undo optimizer-side drift, including decoupled AdamW weight decay."""
        if self.has_frozen:
            self.basis_coeff[self.coeff_mask] = self.frozen_values

    @torch.no_grad()
    def assert_frozen(self):
        if not torch.equal(self.basis_coeff[self.coeff_mask], self.frozen_values):
            raise RuntimeError("Protected WSR coefficients changed during training.")

    @torch.no_grad()
    def to_linear(self):
        self.assert_frozen()
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.basis_coeff.device,
            dtype=self.basis_coeff.dtype,
        )
        linear.weight.copy_(self.basis_coeff @ self.basis.T)
        if self.bias is not None:
            linear.bias.copy_(self.bias)
        return linear


def replace_module(model, name, module):
    parent_name, _, attribute = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, module)


def fold_wsr_modules(model):
    """Remove all WSR wrappers before saving a standard Hugging Face model."""
    names = [
        name for name, module in model.named_modules() if isinstance(module, WSRLinear)
    ]
    for name in names:
        replace_module(model, name, model.get_submodule(name).to_linear())
    return model
