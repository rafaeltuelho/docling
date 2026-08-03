# Cursor demo runbook

## Before the interview

1. Open the repository root in Cursor on
   `agent/docling-contribution-workflow`.
2. Confirm `git status --short --branch` shows only the state you intend to demo.
3. Confirm the development environment can run one fast targeted Docling test.
4. Keep GPU and model-download validation outside the live critical path.

## 1. Plan without editing code

Start a new Cursor Agent conversation with:

> Read `@demo/deepseek-ocr/task-brief.md` and use
> `@.agents/skills/docling-contribution-workflow/SKILL.md`. Run only the Plan
> stage. Inspect current Docling code and tests, then write the proposed plan to
> `@.plans/active/deepseek-ocr.md`. Do not edit product code.

Review the proposed architecture, file boundary, tests, and unresolved questions.
Ask Cursor to revise the plan if needed.

## 2. Approve the plan

Manually change the plan status to `Approved` and complete the human approval
record. This visible checkpoint is part of the solution, not ceremony.

## 3. Implement the approved slice

Continue with:

> Use the Docling contribution workflow to run only the Implement stage for the
> approved plan. Stay within its file boundary. Stop if new scope or an
> architectural decision is required.

Inspect the diff before proceeding.

## 4. Validate and review

Continue with:

> Run the Validate and Review stages. Start with targeted tests, then run the
> relevant repository-native checks. Report exact results, including skips and
> anything the environment cannot run. Fix only findings within the approved
> scope.

Show the terminal output and one repository-specific review finding.

## 5. Prepare the stakeholder handoff

Finish with:

> Run the Hand off stage and complete
> `@demo/deepseek-ocr/change-report.md` from the observed diff and validation
> evidence. Do not claim checks that did not run.

Use the report to close the live flow across contributor, maintainer, QA, PM,
and DevOps concerns.

## Recovery path

If a live agent run becomes slow or expands scope, stop it and resume from the
last reviewed artifact. The demo is the governed workflow and its evidence, not
an uninterrupted autonomous run.
