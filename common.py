"""Self-contained Chinese corpus, tokenizer, and causal-LM batches."""

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SEED = 42
CONTEXT = 128
FACTS = (
    ("太阳", "恒星"), ("月球", "卫星"), ("地球", "行星"),
    ("火星", "行星"), ("木星", "行星"), ("金星", "行星"),
    ("天狼星", "恒星"), ("北极星", "恒星"),
)
TEXT_TRAIN = ("{name}是{kind}。", "天文学里，{name}属于{kind}。", "观察{kind}，例如{name}。")
TEXT_EVAL = ("在分类中，{name}属于{kind}。", "我们观察的{name}是一种{kind}。")
SFT_TRAIN = ("{name}属于哪一类？", "请说出{name}的类型。", "{name}是什么？")
SFT_EVAL = ("请分类：{name}。", "{name}的类别是什么？")


def corpus(evaluate=False):
    templates = TEXT_EVAL if evaluate else TEXT_TRAIN
    return [template.format(name=name, kind=kind) for template in templates for name, kind in FACTS]


def instructions(evaluate=False):
    templates = SFT_EVAL if evaluate else SFT_TRAIN
    return [(template.format(name=name), kind) for template in templates for name, kind in FACTS]


def train_tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        corpus() + [prompt(question) + answer for question, answer in instructions()],
        trainer=trainers.BpeTrainer(
            vocab_size=384,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
            show_progress=False,
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>", model_max_length=CONTEXT,
    )


def prompt(question):
    return f"问题：{question}\n答案："


def example(tokenizer, question, answer, sft):
    prefix = [tokenizer.bos_token_id]
    if sft:
        prefix += tokenizer.encode(prompt(question), add_special_tokens=False)
    targets = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prefix + targets
    if len(ids) > CONTEXT:
        raise ValueError(f"Example has {len(ids)} tokens, exceeding context {CONTEXT}")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + targets}


def examples(tokenizer, sft=False, evaluate=False):
    records = instructions(evaluate) if sft else [("", text) for text in corpus(evaluate)]
    return [example(tokenizer, question, answer, sft) for question, answer in records]


class Collator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, rows):
        width = max(len(row["input_ids"]) for row in rows)
        ids, labels, masks = [], [], []
        for row in rows:
            length = len(row["input_ids"])
            padding = width - length
            ids.append(row["input_ids"] + [self.pad_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            masks.append([1] * length + [0] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }


@torch.inference_mode()
def greedy(model, tokenizer, text, max_new_tokens=12):
    model.eval()
    device = model.get_input_embeddings().weight.device
    ids = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
    if len(ids) + max_new_tokens > CONTEXT:
        raise ValueError("Prompt and continuation exceed model context")
    tokens = torch.tensor([ids], device=device)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=tokens).logits
        next_id = logits[:, -1:].argmax(dim=-1)
        token = next_id.item()
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        tokens = torch.cat((tokens, next_id), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def generations(model, tokenizer, sft):
    if not sft:
        return {"prompt": "天文学里，", "continuation": greedy(model, tokenizer, "天文学里，")}
    rows = [
        {"question": question, "expected": answer, "generated": greedy(model, tokenizer, prompt(question))}
        for question, answer in instructions(evaluate=True)
    ]
    return {"exact_matches": sum(row["expected"] == row["generated"] for row in rows),
            "total": len(rows), "examples": rows}
