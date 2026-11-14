# Security policy

## Supported versions

During 0.x, only the latest minor release receives security fixes.

| Version | Supported |
|---|---|
| 0.0.x (latest) | Yes |

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting:
**Security > Report a vulnerability** on this repository, or
<https://github.com/inboxpraveen/tailtune/security/advisories/new>.

Never open a public issue, discussion or pull request for a vulnerability.

If you cannot use GitHub, email **inboxpraveen.17@gmail.com**.

## What to expect

| Step | Target |
|---|---|
| Acknowledgement | Within 7 days |
| Assessment | Within 14 days |
| Fix or mitigation for high-severity issues | Within 30 days |
| Coordinated disclosure | Within 90 days at most |

Advisories are published through GitHub Security Advisories, with credit to the reporter
unless they ask otherwise.

## In scope

- The release pipeline: workflows, Trusted Publishing and the published packages
- Code execution through model, dataset or config loading
- Crafted spreadsheet or document input
- Escape from the evaluation sandbox
- Path traversal in run directories
- The GGUF conversion environment

## Out of scope

- Bugs in upstream projects. Report them upstream; tailtune tracks them.
- Model behaviour. tailtune makes no safety claim about trained models.

## Security defaults

- **Safetensors only.** tailtune reads and writes safetensors, Parquet, JSON and YAML.
  Pickle checkpoints (`.bin`, `.pt`, `.pth`, `.ckpt`) are refused (error TT-M004), and
  tailtune's own code is linted against `pickle` and `torch.load`.
- **No remote code by default.** Models that need `trust_remote_code` are refused
  (error TT-M003). Enabling it will require naming the repository and a full commit hash.
- **No telemetry.** tailtune sends nothing about its users. Its only network calls are
  the Hugging Face Hub requests you ask for, such as reading a model's `config.json`.
  `tailtune doctor` makes no network calls at all.
- **No secrets in CI for untrusted code.** Workflows never run pull-request code with
  secrets, and every action is pinned to a full commit SHA.
- **No PyPI tokens.** Releases publish through Trusted Publishing behind a
  reviewer-gated `pypi` environment, with attestations.
