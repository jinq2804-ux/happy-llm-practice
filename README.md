# Happy-LLM 第 5、6 章实践

用一套可在本地复现的中文天文分类数据，走完从训练分词器、手写 Llama 风格模型，到 Transformers 训练、参数高效微调和偏好对齐的流程。

| 教材章节 | 本仓库入口 | 实践内容 |
| --- | --- | --- |
| 5.1–5.3 | `chapter5.py`、`model.py` | RMSNorm、RoPE、GQA、SwiGLU、字节级 BPE、PyTorch 预训练和 SFT |
| 6.1–6.3 | `chapter6.py` | Transformers Trainer 预训练、全量 SFT、PEFT LoRA，以及可选的 CUDA QLoRA 和 DeepSpeed ZeRO-2 |
| 6.4 | `chapter6_dpo.py` | 从全量 SFT checkpoint 加载策略与冻结参考模型，使用偏好对训练并记录验证结果 |

课程来源：[Happy-LLM 第 5 章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter5)、[Happy-LLM 第 6 章](https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter6)。

## CPU 复现

在仓库根目录用 Python 3.11 执行。训练只需要本仓库生成的小语料和随机初始化的模型。每个主训练阶段默认 100 个优化步，步数可通过 `--pretrain-steps` 和 `--sft-steps` 调整。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python chapter5.py
python chapter6.py
python chapter6_dpo.py
```

生成文件位于 `runs/`。第 5 章保存 tokenizer、模型配置和两个 PyTorch checkpoint。第 6 章保存 Hugging Face 基座及全量 SFT 模型、LoRA adapter 和 DPO 模型。每个入口都会重载 checkpoint 后执行生成，并把指标和原始生成文本写入 `metrics.json`。

本次实测见 [实验记录](EXPERIMENTS.md) 和 [原始指标](results/)。

## 可选 CUDA 路径

以下命令面向带 NVIDIA GPU 的 Linux 环境。在该环境中安装兼容 CUDA 的 PyTorch 和相应扩展后运行。两个实验各自写入独立目录，使用单进程。本次使用 macOS CPU，以下路径尚未实跑。

```bash
python -m pip install bitsandbytes
python chapter6.py --device cuda --qlora --output runs/chapter6-qlora

python -m pip install deepspeed
python chapter6.py --device cuda --deepspeed deepspeed-zero2.json --output runs/chapter6-zero2
```

QLoRA 从同一个预训练 checkpoint 加载 NF4 量化基座，训练 LoRA adapter。DeepSpeed 配置只用于预训练阶段。
