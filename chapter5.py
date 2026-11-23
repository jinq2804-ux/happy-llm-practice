"""Run tokenizer training, PyTorch pretraining, and response-only SFT."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from common import Collator, SEED, examples, generations, train_tokenizer
from model import Config, TinyLlama


@torch.inference_mode()
def evaluate(model, batches):
    model.eval()
    total_loss = token_count = 0
    for batch in batches:
        count = (batch["labels"][:, 1:] != -100).sum().item()
        total_loss += model(**batch).loss.item() * count
        token_count += count
    return total_loss / token_count


def train_stage(model, tokenizer, sft, steps, rate, output):
    train = examples(tokenizer, sft=sft)
    valid = examples(tokenizer, sft=sft, evaluate=True)
    collate = Collator(tokenizer)
    batches = DataLoader(train, batch_size=8, shuffle=True, collate_fn=collate,
                         generator=torch.Generator().manual_seed(SEED))
    validation = DataLoader(valid, batch_size=8, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate)
    losses = {"eval_before": evaluate(model, validation)}
    model.train()
    step = 0
    while step < steps:
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % 25 == 0:
                print(f"{output.name} step {step}: loss={loss.item():.4f}", flush=True)
            if step == steps:
                break
    losses.update(eval_after=evaluate(model, validation), steps=step, learning_rate=rate,
                  train_examples=len(train), eval_examples=len(valid))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/chapter5"))
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--sft-steps", type=int, default=100)
    args = parser.parse_args()
    if min(args.pretrain_steps, args.sft_steps) < 1:
        parser.error("Step counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = train_tokenizer()
    tokenizer.save_pretrained(args.output)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.output)
    probe = "中文分词测试：月球🌙"
    tokens = tokenizer.tokenize(probe)
    assert tokenizer.decode(tokenizer.encode(probe)) == probe

    config = Config(vocab_size=len(tokenizer))
    (args.output / "config.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
    model = TinyLlama(config)
    with torch.inference_mode():
        ids = torch.tensor([[1, 5, 6, 7]])
        changed = torch.tensor([[1, 5, 8, 9]])
        torch.testing.assert_close(model(ids).logits[:, :2], model(changed).logits[:, :2])
    metrics = {"seed": SEED, "vocab_size": len(tokenizer), "roundtrip": probe,
               "probe_tokens": tokens, "causal_prefix_check": True,
               "parameters": sum(parameter.numel() for parameter in model.parameters())}

    for name, sft, steps, rate in (
        ("pretrain", False, args.pretrain_steps, 3e-3),
        ("sft", True, args.sft_steps, 2e-3),
    ):
        checkpoint = args.output / name
        metrics[name] = train_stage(model, tokenizer, sft, steps, rate, checkpoint)
        model.load_state_dict(torch.load(checkpoint / "model.pt", weights_only=True))
        metrics[name]["generation"] = generations(model, tokenizer, sft)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
