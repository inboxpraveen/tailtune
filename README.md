<h1 align="center">tailtune</h1>
<p align="center"><b>Fine-tune LLMs without forgetting.</b></p>

<p align="center">
  <a href="https://pypi.org/project/tailtune/"><img src="https://img.shields.io/pypi/v/tailtune" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
</p>

Standard fine-tuning teaches a model a new task and quietly breaks what it already knew.
**tailtune** fixes where the update lands: it freezes the top of each weight matrix's
singular spectrum and trains only the low-energy tail. Your model learns the new domain
and keeps its coding, reasoning and instruction-following.

- **Drop-in**: one wrapper around HF Trainer or TRL's SFTTrainer
- **Optimizer-agnostic**: projects the applied update, so it works with AdamW, Muon and others
- **Scales**: single GPU to multi-node with FSDP and DeepSpeed ZeRO-3
- **Methods**: SST (Subspace Tuning) and OSF/OSFT, with more to come

```bash
pip install tailtune
```
