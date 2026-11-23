"""Optional §6.4 experiment: prefer a correct answer over a wrong answer."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, set_seed

from common import Collator, FACTS, SEED, example, generations, instructions

BETA = 0.1


def preference_batches(tokenizer, evaluate=False):
    collate = Collator(tokenizer)
    kinds = sorted({kind for _, kind in FACTS})
    pairs = [
        (example(tokenizer, question, answer, True), example(tokenizer, question, wrong, True))
        for question, answer in instructions(evaluate)
        for wrong in kinds if wrong != answer
    ]

    def collate_pairs(rows):
        chosen, rejected = zip(*rows)
        return collate(chosen), collate(rejected)

    return DataLoader(
        pairs, batch_size=8, shuffle=not evaluate, collate_fn=collate_pairs,
        generator=torch.Generator().manual_seed(SEED),
    )


def response_logprob(model, batch):
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    token_logprobs = logits[:, :-1].log_softmax(-1)
    selected = token_logprobs.gather(-1, labels.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~valid, 0).sum(-1)


def preference_margin(model, chosen, rejected):
    return response_logprob(model, chosen) - response_logprob(model, rejected)


@torch.inference_mode()
def evaluate(policy, reference, batches):
    policy.eval()
    margins = []
    for chosen, rejected in batches:
        difference = preference_margin(policy, chosen, rejected)
        margins.append(difference - preference_margin(reference, chosen, rejected))
    margins = torch.cat(margins)
    return {"loss": -F.logsigmoid(BETA * margins).mean().item(),
            "reward_margin": (BETA * margins).mean().item(),
            "positive_margins": margins.gt(0).sum().item(), "pairs": margins.numel()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/chapter6"))
    parser.add_argument("--output", type=Path, default=Path("runs/chapter6-dpo"))
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epoch count must be positive")
    torch.set_num_threads(2)
    set_seed(SEED)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.input)
    checkpoint = args.input / "full_sft"
    policy = LlamaForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    reference = LlamaForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    reference.eval().requires_grad_(False)
    training = preference_batches(tokenizer)
    validation = preference_batches(tokenizer, evaluate=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = {"seed": SEED, "beta": BETA, "steps": args.epochs * len(training),
               "training_pairs": len(training.dataset),
               "before": evaluate(policy, reference, validation)}
    policy.train()
    for _ in range(args.epochs):
        for chosen, rejected in training:
            with torch.no_grad():
                reference_margin = preference_margin(reference, chosen, rejected)
            margin = preference_margin(policy, chosen, rejected) - reference_margin
            loss = -F.logsigmoid(BETA * margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
    policy.save_pretrained(args.output)
    restored = LlamaForCausalLM.from_pretrained(args.output, local_files_only=True)
    metrics["after"] = evaluate(restored, reference, validation)
    metrics["generation"] = generations(restored, tokenizer, True)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
