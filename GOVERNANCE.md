# Governance

tailtune runs as a BDFL project (one named lead with the final say) with written
policies. This document changes only through an RFC.

## Roles

- **BDFL:** Praveen Kumar ([@inboxpraveen](https://github.com/inboxpraveen)), the
  project owner.
- **Maintainers:** merge, review and release rights. Listed in
  [MAINTAINERS.md](MAINTAINERS.md).
- **Contributors:** everyone else who takes part.

## Decisions

- **Lazy consensus on pull requests.** A non-trivial pull request with no objection
  after 72 hours may merge.
- **Disagreements are discussed in public.** The BDFL decides and writes down the
  reason.
- **Locked decisions** (D-01 to D-25 in the [plan index](docs/plan/README.md)) change
  only through a gate result or an owner decision, recorded in the index first.

## RFCs

An RFC is needed for a change to a locked decision, a breaking change to the public
API or config schema, a new method or new required dependency, telemetry, or
licensing, governance or sponsor policy.

1. Open a Discussion in the RFC category: problem, proposal, evidence with labels,
   alternatives, migration, and who will maintain it.
2. Comments stay open for 14 days, or 30 for telemetry and licensing.
3. The BDFL decides and records the outcome.
4. An accepted RFC is committed as `rfcs/NNNN-title.md`.

## Becoming a maintainer

- About 10 merged non-trivial pull requests over at least 3 months
- Regular reviews of other people's pull requests
- Ownership of an area, such as data ingestion, evaluation or a backend
- Following the [Code of Conduct](CODE_OF_CONDUCT.md)

A maintainer nominates, and the BDFL confirms. Maintainers who are inactive for
6 months move to emeritus status and their access is removed.

## Continuity

- Once a second maintainer exists, they become co-owner of the PyPI project, the
  GitHub and Hugging Face organisations and the domain.
- If the BDFL is unreachable for 60 days, the maintainers may act on the project's
  behalf.
- Credentials and recovery steps live in a private runbook held by the maintainers.
- The project commits to 12 months of maintenance from each minor release.

## Money and conflicts of interest

- No CLA. Contributions are accepted under Apache-2.0 with a DCO sign-off, so they
  cannot be relicensed without their authors.
- Sponsors are disclosed beside any benchmark that features their hardware, model
  or product.
- Benchmarks are preregistered in a dated commit before any runs. Sponsors see no
  results early, cannot request reruns or exclusions, and placement is never sold.
- Maintainers list their employment and any paid relationship with vendors in
  [MAINTAINERS.md](MAINTAINERS.md), and step back from choosing settings for
  benchmark arms involving those vendors.

## Changing governance

Through an RFC. The project moves to a small steering group once it has 3 or more
maintainers.
