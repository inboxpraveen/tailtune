<h1 align="center">tailtune</h1>
<p align="center"><b>Fine-tune LLMs with far less forgetting, and prove it.</b></p>

<p align="center">
  <a href="https://pypi.org/project/tailtune/"><img src="https://img.shields.io/pypi/v/tailtune" alt="PyPI"></a>
  <a href="https://github.com/inboxpraveen/tailtune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/status-pre--alpha-orange" alt="Status">
</p>

Fine-tuning teaches a model something new and can quietly damage what it
already knew. tailtune is being built to show you exactly what changed on
every run, and to reduce the damage as far as your hardware allows.

**Status: pre-alpha.** Only the two hardware commands below work today.
Training is not built yet. Follow the [implementation plan](https://github.com/inboxpraveen/tailtune/blob/main/docs/plan/README.md)
to see what is being built and why.

## Available now (0.0.1)

```bash
pip install tailtune
tailtune doctor                    # is this machine ready to fine-tune?
tailtune estimate Qwen/Qwen3-8B    # will this model fit, and with which method?
```

`doctor` checks Python, torch, your GPU and driver, real bf16 support, memory and
disk. `estimate` reads only the model's `config.json`, never its weights. It prints
the GPU and host memory each method needs, whether it fits this machine or a card
you name with `--gpu`, and which method tailtune would pick. Its figures are
computed from the model's shapes and have not yet been checked against real
hardware.

## What we are building

- **Start from what you have.** Excel read as a real table, plus CSV, JSON,
  Parquet and Hugging Face datasets, with a validation report before training.
- **A forgetting report on every run.** Before-and-after scores on coding,
  instruction following, maths, knowledge, languages and refusals.
  It will also work on a model trained with any other tool.
- **Protection that fits your hardware.** LoRA on small GPUs. SST
  (Chavan et al., 2026) and OSF where they fit. The tool tells you which
  applies before it loads any weights.
- **A running model at the end.** Merged safetensors, GGUF and Ollama.
