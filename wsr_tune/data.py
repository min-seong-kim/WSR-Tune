"""Safety calibration and downstream task data with response-only labels.

Downstream tasks are loaded from Hugging Face Datasets (or a local JSON/JSONL
copy with the same row schema) and converted to (prompt, response) examples.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)

QA_PROMPT = "Question: {question}\nAnswer:"


@dataclass(frozen=True)
class Example:
    """One supervised pair.

    ``prompt`` is the user turn for the chat format, ``response`` is the
    supervised target, and ``plain_prompt`` is the prompt text used when the
    tokenizer has no chat template (``--chat_format plain``).
    """

    prompt: str
    response: str
    plain_prompt: str


def qa_example(question, response):
    question, response = question.strip(), response.strip()
    return Example(question, response, QA_PROMPT.format(question=question))


def read_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a nonempty list of examples.")
    return rows


# --------------------------------------------------------------------------- #
# Safety calibration data (Phases 1 and 2)
# --------------------------------------------------------------------------- #


def read_safety_pairs(path, max_samples=0):
    rows = read_rows(path)
    if max_samples > 0:
        rows = rows[:max_samples]
    examples = []
    for index, row in enumerate(rows):
        # Only the safe refusal (`llama3_output`) is ever used as a target.
        prompt, response = row.get("prompt"), row.get("llama3_output")
        if not all(isinstance(x, str) and x.strip() for x in (prompt, response)):
            raise ValueError(
                f"Safety row {index} needs nonempty prompt and llama3_output fields."
            )
        examples.append(qa_example(prompt, response))
    return examples


# --------------------------------------------------------------------------- #
# Downstream tasks (Phase 3)
# --------------------------------------------------------------------------- #

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
MATH_BLANK_LINES = re.compile(r"\n{3,}")

ARC_PROMPT = (
    "Given the following question and four candidate answers (A, B, C and D), "
    "choose the best answer.\n"
    "Question: {question}\n"
    "{choices}\n"
    'Your response should end with "The best answer is [the_answer_letter]" '
    "where the [the_answer_letter] is one of A, B, C or D."
)
ARC_ANSWER_PREFIX = "The best answer is"
ARC_LETTERS = {"1": "A", "2": "B", "3": "C", "4": "D"}

MEDQA_INSTRUCTION = (
    "Answer the following multiple-choice medical question by selecting the "
    "single best answer. Reply with only the option letter (A, B, C, or D) "
    "followed by a period and the answer text.\n"
    "Example: A. Aspirin"
)


def _text(value):
    return "" if value is None else str(value).strip()


def _gsm8k_example(row):
    question, answer = _text(row.get("question")), _text(row.get("answer"))
    if not question or not answer:
        return None
    return qa_example(question, answer)


def _last_boxed(text):
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) span, or None."""
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _strip_box(span):
    if span.startswith("\\boxed "):
        return span[len("\\boxed ") :]
    for left in ("\\boxed{", "\\fbox{"):
        if span.startswith(left) and span.endswith("}"):
            return span[len(left) : -1]
    return span


def math_target(solution):
    """Rewrite a MATH solution as a rationale followed by ``Final Answer: $x$``."""
    solution = solution.strip()
    boxed = _last_boxed(solution)
    if boxed is None:
        return None
    final = _strip_box(boxed).strip()
    rationale = solution.replace(boxed, final)
    for token in ("$", "\\[", "\\]", "\\(", "\\)", "\\boxed", "\\fbox"):
        rationale = rationale.replace(token, "")
    rationale = MATH_BLANK_LINES.sub("\n\n", rationale).strip()
    return f"{rationale}\nFinal Answer: ${final}$"


def _math_example(row):
    problem, solution = _text(row.get("problem")), _text(row.get("solution"))
    target = math_target(solution) if problem and solution else None
    if target is None:
        return None
    return qa_example(problem, target)


def _arc_example(row):
    question, choices = _text(row.get("question")), row.get("choices") or {}
    labels, texts = choices.get("label") or [], choices.get("text") or []
    answer = ARC_LETTERS.get(str(row.get("answerKey")), _text(row.get("answerKey")))
    if not question or not labels or len(labels) != len(texts) or not answer:
        return None
    lines = [
        f"{ARC_LETTERS.get(str(label), str(label))}. {_text(text)}"
        for label, text in zip(labels, texts)
    ]
    prompt = ARC_PROMPT.format(question=question, choices="\n".join(lines))
    return Example(prompt, f"{ARC_ANSWER_PREFIX} {answer}", prompt + "\n")


def _medqa_example(row):
    data = row.get("data") or row  # nested HF record or flattened fields
    question = _text(data.get("Question"))
    options = data.get("Options") or {}
    option, answer = _text(data.get("Correct Option")), _text(data.get("Correct Answer"))
    if not question or not options or not option or not answer:
        return None
    option_lines = "\n".join(f"{k}. {_text(v)}" for k, v in sorted(options.items()))
    input_text = f"{question}\n\n{option_lines}"
    plain_prompt = (
        f"### Instruction:\n{MEDQA_INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        "### Response:\n"
    )
    return Example(
        f"{MEDQA_INSTRUCTION}\n\n{input_text}", f"{option}. {answer}", plain_prompt
    )


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source: str
    formatter: object
    shuffle: str  # "never", "always", or "subsample" (only when truncating)


TASKS = {
    "gsm8k": TaskSpec("GSM8K", "openai/gsm8k (main, train)", _gsm8k_example, "never"),
    "math": TaskSpec(
        "Hendrycks MATH",
        "EleutherAI/hendrycks_math (all seven subjects, train)",
        _math_example,
        "always",
    ),
    "arc_challenge": TaskSpec(
        "ARC-Challenge", "allenai/ai2_arc (ARC-Challenge, train)", _arc_example, "never"
    ),
    "medqa": TaskSpec(
        "MedQA (USMLE)", "openlifescienceai/medqa (train)", _medqa_example, "subsample"
    ),
}
TASK_NAMES = tuple(TASKS)


def load_task_rows(task, path=None):
    """Return a `datasets.Dataset` of raw rows for a task."""
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets, load_dataset

    if path:
        rows = read_rows(path)
        if task == "medqa":
            # Accept the nested Hugging Face record or its flattened fields.
            rows = [row.get("data") or row for row in rows]
        return HFDataset.from_list(rows)
    if task == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split="train")
    if task == "math":
        return concatenate_datasets(
            [
                load_dataset("EleutherAI/hendrycks_math", subject, split="train")
                for subject in MATH_SUBJECTS
            ]
        )
    if task == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    if task == "medqa":
        return load_dataset("openlifescienceai/medqa", split="train")
    raise ValueError(f"Unknown task {task!r}; choose from {TASK_NAMES}.")


def read_task_examples(task, path=None, max_samples=0, seed=42):
    """Load, select, and format downstream training examples for a task.

    GSM8K and ARC-Challenge take the first ``max_samples`` rows. MATH is
    shuffled with ``seed`` before selection. MedQA is shuffled with ``seed``
    only when ``max_samples`` truncates it.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {TASK_NAMES}.")
    spec = TASKS[task]
    rows = load_task_rows(task, path)
    if spec.shuffle == "always":
        rows = rows.shuffle(seed=seed)
    if max_samples > 0 and len(rows) > max_samples:
        if spec.shuffle == "subsample":
            rows = rows.shuffle(seed=seed)
        rows = rows.select(range(max_samples))
    examples, skipped = [], 0
    for row in rows:
        example = spec.formatter(dict(row))
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    if skipped:
        LOGGER.warning("Skipped %d %s rows with missing fields.", skipped, spec.name)
    if not examples:
        raise ValueError(f"No usable {spec.name} examples were found.")
    LOGGER.info(
        "Loaded %d %s examples from %s.", len(examples), spec.name, path or spec.source
    )
    return examples


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #


def resolve_chat_format(tokenizer, requested):
    if requested == "auto":
        return "chat" if tokenizer.chat_template else "plain"
    if requested == "chat" and not tokenizer.chat_template:
        raise ValueError("chat format requires a tokenizer with a chat template.")
    return requested


def format_pair(tokenizer, example, chat_format):
    """Return (prompt_text, full_text) for an example."""
    if chat_format == "chat":
        messages = [{"role": "user", "content": example.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": example.response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt, full
    prompt = example.plain_prompt
    separator = "" if prompt.endswith(("\n", " ")) else " "
    return prompt, f"{prompt}{separator}{example.response}"


def tokenize_pair(tokenizer, example, max_length, chat_format, calibration=False):
    """Tokenize one example with labels only on the response tokens.

    Calibration examples (Phases 1 and 2) are tokenized as a single text with
    the tokenizer's special tokens. Downstream examples (Phase 3) are tokenized
    without added special tokens; in the plain format the response is tokenized
    separately and terminated with EOS when it fits.
    """
    prompt, full = format_pair(tokenizer, example, chat_format)
    kwargs = dict(
        truncation=True, max_length=max_length, add_special_tokens=calibration
    )
    prompt_ids = tokenizer(prompt, **kwargs)["input_ids"]
    if not calibration and chat_format == "plain":
        remaining = max_length - len(prompt_ids)
        answer_ids = tokenizer(
            example.response,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, remaining),
        )["input_ids"][:remaining]
        eos = tokenizer.eos_token_id
        if eos is not None and len(answer_ids) < remaining:
            if not answer_ids or answer_ids[-1] != eos:
                answer_ids.append(eos)
        ids = prompt_ids + answer_ids
    else:
        ids = tokenizer(full, **kwargs)["input_ids"]
    prompt_length = min(len(prompt_ids), len(ids))
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * prompt_length + ids[prompt_length:],
    }


class PairDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        max_length,
        chat_format,
        calibration=False,
        require_response=True,
    ):
        self.examples = []
        dropped = 0
        for example in examples:
            encoded = tokenize_pair(
                tokenizer, example, max_length, chat_format, calibration
            )
            if require_response and not any(x != -100 for x in encoded["labels"][1:]):
                dropped += 1
                continue
            self.examples.append(encoded)
        if dropped:
            LOGGER.warning(
                "Dropped %d examples with no response tokens after truncation.", dropped
            )
        if not self.examples:
            raise ValueError(
                "No usable examples; increase max_length or check the data."
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@dataclass
class ResponseCollator:
    pad_token_id: int

    def __call__(self, examples):
        length = max(len(example["input_ids"]) for example in examples)
        padded = {"input_ids": [], "attention_mask": [], "labels": []}
        fill = {"input_ids": self.pad_token_id, "attention_mask": 0, "labels": -100}
        for example in examples:
            for key in padded:
                padded[key].append(
                    example[key] + [fill[key]] * (length - len(example[key]))
                )
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()
        }
