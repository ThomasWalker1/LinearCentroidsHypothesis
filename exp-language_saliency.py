"""Generate image-only, answer-targeted local-centroid saliency maps for a causal LM.

The centroid is computed with respect to the continuous input embedding
sequence, because token IDs themselves are discrete.  For a supplied,
teacher-forced answer, the target is the sum of its causal log probabilities.
Each output PNG overlays the local-centroid intensity on the tokenized
``Question: ... Answer: ...`` sentence.

Examples:
    python exp-language_saliency.py
    python exp-language_saliency.py --model gpt2 --samples 24
    python exp-language_saliency.py --examples my_questions.json

``--examples`` accepts a JSON list of {"question": "...", "answer": "..."}
objects.  The experiment writes PNGs only; it does not create HTML or JSON
outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_EXAMPLES = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "What is two plus two?", "answer": "4"},
    {"question": "What is the chemical formula for water?", "answer": "H2O"},
    {"question": "What color is a clear daytime sky?", "answer": "blue"},
    {"question": "What is the opposite of hot?", "answer": "cold"},
    {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
]


def question_answer_ids(tokenizer, question: str, answer: str) -> tuple[torch.Tensor, int]:
    """Encode a prompt and answer separately, retaining the answer boundary."""
    prompt = f"Question: {question}\nAnswer:"
    answer_text = answer if answer.startswith((" ", "\n", "\t")) else f" {answer}"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids or not answer_ids:
        raise ValueError("Both the question prompt and answer must encode to at least one token.")
    return torch.tensor([prompt_ids + answer_ids], dtype=torch.long), len(prompt_ids)


def answer_targeted_local_centroid(
    model,
    input_ids: torch.Tensor,
    answer_start: int,
    n_samples: int,
    sigma: float,
    generator: torch.Generator,
    target_answer_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, list[int]]:
    """Average answer-log-probability gradients in an embedding neighbourhood.

    The first neighbourhood member is exactly the original embedding.  At each
    token position the displayed intensity is the L2 norm of the resulting
    signed centroid vector.
    """
    if n_samples < 1 or sigma < 0:
        raise ValueError("n_samples must be positive and sigma must be non-negative.")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    sequence_length = input_ids.shape[1]
    answer_length = sequence_length - answer_start
    if answer_start < 1 or answer_length < 1:
        raise ValueError("The answer must follow at least one prompt token.")

    selected = list(range(answer_length)) if target_answer_indices is None else list(target_answer_indices)
    if not selected or min(selected) < 0 or max(selected) >= answer_length:
        raise ValueError("Target answer indices must identify answer tokens.")

    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)
    scale = embeddings.detach().float().std().clamp_min(torch.finfo(torch.float32).eps)
    noise = torch.randn(
        (n_samples,) + tuple(embeddings.shape[1:]), device=device,
        dtype=embeddings.dtype, generator=generator,
    ) * (sigma * scale.to(embeddings.dtype))
    noise[0].zero_()
    neighbourhood = (embeddings.expand(n_samples, -1, -1) + noise).detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(
        inputs_embeds=neighbourhood,
        attention_mask=torch.ones((n_samples, sequence_length), dtype=torch.long, device=device),
        use_cache=False,
    )
    prediction_positions = torch.tensor([answer_start + index - 1 for index in selected], device=device)
    target_ids = input_ids[0, [answer_start + index for index in selected]]
    target_logits = outputs.logits[:, prediction_positions, :]
    target_log_probs = F.log_softmax(target_logits.float(), dim=-1).gather(
        -1, target_ids.view(1, -1, 1).expand(n_samples, -1, -1)
    ).squeeze(-1)
    target_log_probs.sum().backward()

    centroid = neighbourhood.grad.detach().mean(dim=0).cpu()
    scores = centroid.float().norm(dim=-1)
    return centroid, scores, target_log_probs.detach().mean().item(), selected


def _safe_stem(index: int, question: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    return f"{index:02d}_{slug[:60]}"


def _render_saliency_image(
    tokenizer,
    input_ids: torch.Tensor,
    answer_start: int,
    scores: torch.Tensor,
    selected_answer_indices: Sequence[int],
    title: str,
    subtitle: str,
    save_path: Path,
) -> None:
    """Overlay individually normalized token intensities on a wrapped sentence."""
    ids = input_ids[0].tolist()
    values = scores.tolist()
    max_value = max(values) if values else 0.0
    denominator = max(max_value, 1e-12)
    targets = {answer_start + index for index in selected_answer_indices}

    fig, ax = plt.subplots(figsize=(12, 3.4), dpi=180)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.72)
    fig.suptitle(title, x=0.03, y=0.96, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.03, 0.86, subtitle, ha="left", va="top", fontsize=9, color="#374151")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_width = ax.get_window_extent(renderer).width

    x, y = 0.01, 0.82
    line_height = 0.26
    for position, (token_id, value) in enumerate(zip(ids, values)):
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        # GPT-style tokenizers represent the prompt/answer separator as an
        # actual newline.  It has zero visual width, so handle it as a layout
        # instruction instead of drawing it as an invisible span.
        if "\n" in token and not token.replace("\n", "").strip():
            x, y = 0.01, y - line_height
            continue
        measure = ax.text(x, y, token, transform=ax.transAxes, fontsize=12, family="DejaVu Sans", alpha=0)
        width = measure.get_window_extent(renderer).width / axes_width
        measure.remove()
        if x + width > 0.99 and x > 0.01:
            x, y = 0.01, y - line_height

        intensity = value / denominator
        facecolor = (0.86, 0.15, 0.15, 0.06 + 0.88 * intensity)
        is_answer = position >= answer_start
        is_target = position in targets
        edgecolor = "#1d4ed8" if is_target else ("#60a5fa" if is_answer else "none")
        linewidth = 1.6 if is_target else (0.8 if is_answer else 0.0)
        ax.text(
            x, y, token, transform=ax.transAxes, fontsize=12, family="DejaVu Sans", va="center",
            bbox={"facecolor": facecolor, "edgecolor": edgecolor, "linewidth": linewidth, "boxstyle": "round,pad=0.12"},
        )
        x += width

    fig.text(0.03, 0.035, "Red intensity: local-centroid L2 saliency. Blue outline: answer target token(s).", fontsize=8, color="#4b5563")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_examples(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_EXAMPLES
    examples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("Examples must be a non-empty JSON list.")
    if any(not isinstance(x, dict) or not isinstance(x.get("question"), str) or not isinstance(x.get("answer"), str) for x in examples):
        raise ValueError("Each example needs string 'question' and 'answer' fields.")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="Hugging Face causal-LM identifier.")
    parser.add_argument("--samples", type=int, default=24, help="Embedding-neighbourhood samples per example.")
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise scale relative to embedding standard deviation.")
    parser.add_argument("--examples", type=Path, help="Optional JSON list of question/answer objects.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a torch device such as 'cuda'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/language_saliency/<model>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = _load_examples(args.examples)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    output_dir = args.output_dir or ROOT / "outputs" / "language_saliency" / re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.model)
    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for index, example in enumerate(examples, start=1):
        input_ids, answer_start = question_answer_ids(tokenizer, example["question"], example["answer"])
        _, scores, mean_log_prob, selected = answer_targeted_local_centroid(
            model, input_ids, answer_start, args.samples, args.sigma, generator
        )
        output_path = output_dir / f"{_safe_stem(index, example['question'])}.png"
        _render_saliency_image(
            tokenizer, input_ids, answer_start, scores, selected,
            title=f"{example['question']}  →  {example['answer']}",
            subtitle=(f"{args.model} | {args.samples} local embedding samples | "
                      f"mean answer log-probability: {mean_log_prob:.3f}"),
            save_path=output_path,
        )
        print(f"[{index}/{len(examples)}] wrote {output_path}")


if __name__ == "__main__":
    main()
