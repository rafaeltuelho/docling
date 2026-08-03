---
name: docling-contribution-workflow
description: Plan, implement, validate, review, and hand off a repository-aware Docling contribution. Use for the DeepSeek-OCR demo or whenever a contributor must move a scoped Docling task from requirements to a PR-ready report with a human approval gate.
---

# Docling Contribution Workflow

Use the repository as the source of truth. Historical PR material is evidence,
not a substitute for inspecting current code.

## Required context

Read these files before planning:

- `AGENTS.md`
- `CONTRIBUTING.md`
- `Makefile`
- `pyproject.toml`
- `demo/deepseek-ocr/task-brief.md`
- `demo/deepseek-ocr/maintainer-context.md`

Inspect current source and tests for at least two adjacent implementations. For
the DeepSeek-OCR case, include the current Nemotron OCR integration and at least
one VLM or OCR implementation relevant to the proposed architecture.

## 1. Plan

Do not edit product code or tests during this stage.

1. Inspect `git status`, the current branch, and the base commit.
2. Trace the existing extension points, public options, plugin registration,
   dependency extras, tests, CLI exposure, and documentation.
3. Reconcile the task brief and historical maintainer feedback with current
   architecture. Call out anything that is obsolete or unresolved.
4. Propose the smallest coherent vertical slice. Avoid dead scaffolding that is
   tested only for its own sake.
5. Write the result to `.plans/active/deepseek-ocr.md` using its
   template. Leave its status as `Proposed`.
6. Stop and ask for human approval.

The agent must never mark its own plan `Approved`.

## 2. Implement

Proceed only when the plan status is `Approved` and the approval record names
the human approver.

1. Modify only files listed in the approved change boundary.
2. Prefer current Docling abstractions over code copied from PR #2721.
3. Keep optional dependencies isolated and imports lazy where nearby models do.
4. Keep hardware-independent behavior testable without downloading or loading
   the model.
5. Add meaningful tests for contracts and failure modes. Do not create tests
   that merely restate new implementation details.
6. Update user-facing documentation and dependency metadata only when the slice
   makes them necessary.
7. If implementation requires a new file or architectural decision, update the
   plan and return to human approval before continuing.

## 3. Validate

Run fast, targeted checks first, followed by repository checks appropriate to
the changed files.

1. Run the focused pytest selection for touched behavior.
2. Run `make validate`; review any mutations and rerun until clean.
3. Run `make check` when time and the local environment allow it.
4. Run GPU or end-to-end model validation only when compatible hardware and
   dependencies are present.
5. Record every command and outcome exactly, including failures, skips, and
   validations not run.

Do not turn a skipped hardware test into a passing result.

## 4. Review

Review the final diff against:

- acceptance criteria and approved boundaries;
- current OCR and VLM architecture;
- backward compatibility and public API impact;
- optional dependency and import-time behavior;
- device selection and failure messages;
- test value, coverage gaps, and hardware assumptions;
- documentation and operational impact;
- unrelated edits, generated files, and lockfile churn.

Resolve findings when they stay within the approved plan. Otherwise record them
as follow-up work and request a new approval before expanding scope.

## Plan lifecycle

- Create or update the implementation plan under `.plans/active/`.
- Keep the plan active throughout implementation, validation, and review.
- The agent must not approve its own plan.
- Scope expansion requires updating the plan and returning to human approval.
- Move the plan to `.plans/completed/` only after validation and handoff.
- Move abandoned or superseded plans to `.plans/archived/`.

## 5. Hand off

Complete `demo/deepseek-ocr/change-report.md` from observed evidence. Write for
engineers, maintainers, QA, PM, and DevOps. Keep limitations explicit and make
the report usable as the basis of a pull request description.
