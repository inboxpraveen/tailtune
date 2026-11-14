# Changelog

All notable changes are recorded here. tailtune uses `0.MINOR.PATCH` versioning
during 0.x: patches never break anything, and minor releases break something only
after a deprecation period.

## Unreleased

### Added

- `tailtune doctor`: checks D01 to D18 from the plan (Python, platform, hosted
  notebooks, torch, accelerator and support tier, driver, VRAM, native bf16, host
  RAM, disk, multi-GPU, Transformers, Accelerate, PEFT and TRL against the tested
  ranges, extras, attention backends, MPS, XPU, CPU SVD). Makes no network calls.
  `--json` for machine-readable output, `--strict` to exit 4 on any failed check,
  and `--deep` for a 2-rank gloo test and a bf16 rounding check.
- `tailtune estimate MODEL`: reads `config.json` only, never weights, and prints GPU
  and host memory for every route (`lora-4bit`, `lora-16bit`, `osf`, five `sst`
  routes, `full`), with a -15% / +25% range, a fit verdict for this machine or a
  named card (`--gpu`, `--vram`, `--host-ram`), the smallest single card, and the
  `method=auto` choice. Supports Llama, Mistral, Qwen2, Qwen3 and Qwen3-MoE configs,
  and refuses other architectures, quantized and multimodal checkpoints, pickle-only
  repositories and remote code rather than guess.
- Python API: `tailtune.doctor()` and `tailtune.estimate()`.
- Exit codes 0 to 5 and 130, with catalogued error codes such as `TT-C003`.

All memory figures are computed from shapes and have not been checked against
hardware. Calibration is Gate 1 of the plan.

### Compatibility

| tailtune | Python | torch | Transformers | TRL | PEFT | Notes |
|---|---|---|---|---|---|---|
| 0.0.1 | 3.11 to 3.13 | 2.14, 2.15 probed, not required | probed | probed | probed | `doctor`, `estimate` only |
