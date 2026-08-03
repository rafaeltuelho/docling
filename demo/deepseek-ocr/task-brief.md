# Task brief: architecture-first DeepSeek-OCR contribution

## Customer problem

A contributor wants to add DeepSeek-OCR to Docling. A previous implementation
was technically plausible but discovered important architectural, dependency,
device, and test constraints late in pull-request review. The goal is to move
those discoveries into planning and produce a smaller, safer contribution.

## Desired outcome

Use the Docling contribution workflow to:

1. determine where a DeepSeek-OCR backbone belongs in the current architecture;
2. distinguish reusable model inference from its OCR, VLM, and future
   post-processing usages;
3. select one small, coherent vertical slice for implementation;
4. implement it only after explicit plan approval;
5. validate and review it with repository-native tooling; and
6. create an evidence-based stakeholder handoff.

## Acceptance criteria

- The plan compares current, nearby OCR and VLM implementations, including
  `docling/models/stages/ocr/nemotron_ocr_model.py`.
- The architecture does not couple the reusable DeepSeek model backbone to a
  single usage without an explicit rationale.
- The first slice has a clear user or integration outcome; it is not dead
  scaffolding added only to make unit tests pass.
- MPS support is out of scope until upstream compatibility is demonstrated.
- Hardware-independent policy and parsing behavior can be tested without a GPU,
  network access, remote-code execution, or a model download.
- Hardware-dependent tests detect capabilities and skip with a clear reason.
- Optional dependencies do not break ordinary Docling imports.
- Prompts and output parsing are constrained to formats Docling can support
  reliably; arbitrary prompts are not exposed without a compatible parser.
- Validation results list exact commands, failures, skips, and unrun checks.
- The final report addresses engineering, maintainer, QA, PM, and DevOps needs.

## Constraints

- Preserve backward compatibility unless the approved plan says otherwise.
- Prefer current Docling extension points and directory layout over the paths
  used by historical PR #2721.
- Do not change unrelated formatting, lockfiles, or dependencies.
- Do not use external code or documentation as authoritative unless the human
  explicitly approves that source.
- Do not claim production readiness from mocked or CPU-only tests.

## Non-goals for the first slice

- Complete support for all three DeepSeek-OCR usages.
- Apple Silicon/MPS support.
- GLM-OCR or another model family.
- A live multi-gigabyte model download during the interview.
- A guarantee that maintainers will accept the resulting design.

## Stakeholders

- **Contributor:** clear files, conventions, commands, and decision points.
- **Maintainer:** architecture fit, scope control, compatibility, and reviewability.
- **QA:** meaningful test matrix, edge cases, and explicit coverage gaps.
- **PM:** user-visible outcome, exclusions, risk, and next steps.
- **DevOps:** dependency, hardware, CI, cache, and deployment implications.
