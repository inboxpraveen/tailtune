<h1 align="center">tailtune</h1>
<p align="center"><b>Fine-tune LLMs with far less forgetting, and prove it.</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/status-pre--alpha-orange" alt="Status">
</p>

Fine-tuning teaches a model something new and can quietly damage what it
already knew. tailtune is being built to show you exactly what changed on
every run, and to reduce the damage as far as your hardware allows.

**Status: pre-alpha.** Nothing is ready to use yet. Follow the
[implementation plan](docs/plan/README.md) to see what is being built and why.

What we are building:

- **Start from what you have.** Excel read as a real table, plus CSV, JSON,
  Parquet and Hugging Face datasets, with a validation report before training.
- **A forgetting report on every run.** Before-and-after scores on coding,
  instruction following, maths, knowledge, languages and refusals.
  It will also work on a model trained with any other tool.
- **Protection that fits your hardware.** LoRA on small GPUs. SST
  (Chavan et al., 2026) and OSF where they fit. The tool tells you which
  applies before it loads any weights.
- **A running model at the end.** Merged safetensors, GGUF and Ollama.
