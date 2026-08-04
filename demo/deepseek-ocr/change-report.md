# DeepSeek-OCR contribution report

## Executive summary

- Outcome: Hardened the existing API-only DeepSeek-OCR VLM conversion path
  (`deepseek_ocr`) so grounded-markdown parsing is dependency-safe, failure
  behavior is explicit, and hardware-independent tests run in core CI.
- User-visible behavior: Nonblank VLM responses with zero valid grounded
  annotations now surface as `partial_success` with exactly one
  `INFERENCE_FAILURE`. Blank-but-present predictions remain empty successful
  pages. Unknown grounded labels still map to `TEXT`. Malformed / nonnumeric
  coordinate annotations are skipped with warnings when siblings survive.
- Explicit exclusions: classic OCR adapter, local Transformers/vLLM/MLX
  inference, MPS, post-processing OCR, lockfile/dependency-extra changes, and
  fixing legacy `VlmPipeline` Transformers import coupling.
- Overall readiness: **PR-ready under qualified validation** (live Ollama E2E
  not run; `make check` still fails on pre-existing unrelated max-lines debt).

## Concise change summary

Make DeepSeek-OCR’s API VLM grounded-markdown path reliable without adding a
local model backbone: remove undeclared `lxml` use, add parser diagnostics,
deduplicate incomplete-stop-reason errors with parser-format failures, split
live Ollama tests into an `external_service` module using the current preset,
and document failure semantics.

## Motivation and approved scope

### Motivation

Historical PR #2721 exposed architectural and test gaps late. Current main
already exposes an API-only DeepSeek VLM preset, but the parser depended on
undeclared `lxml`, silently accepted unparseable nonblank output, and mixed
hardware-independent tests under `ml_vlm`.

### Approved scope (human-approved plan)

Vertical slice: **VLM API hardening only** — reusable DeepSeek family stays
conceptual; this slice hardens the existing VLM adapter/parser only.

Approved boundary files:

- `docling/utils/deepseekocr_utils.py`
- `docling/pipeline/vlm_pipeline.py`
- `tests/test_deepseekocr_vlm.py`
- `tests/test_deepseekocr_vlm_external.py` (new)
- `tests/data/md_deepseek/` (no fixture churn required)
- `docs/usage/vision_models.md`
- `docs/usage/model_catalog.md`

Approved by: Rafael Soares (maintainer proxy), 2026-08-04.
Accepted after review correction cycle: Rafael Soares, 2026-08-04.

## Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Plan compared Nemotron OCR and nearby VLM paths | Met | Plan architecture evidence; Nemotron + Granite/DeepSeek VLM |
| Architecture does not couple a backbone to one usage without rationale | Met | API-only slice; backbone deferred with explicit rationale |
| First slice has clear user/integration outcome | Met | Reliable `deepseek_ocr` API conversion contract |
| MPS out of scope | Met | Not implemented; not validated |
| Hardware-independent policy/parsing testable without GPU/network/download | Met | `tests/test_deepseekocr_vlm.py` — 16 passed |
| Hardware-dependent tests capability-detect and skip clearly | Met | External module skips: Ollama unavailable / required model unavailable |
| Optional deps do not break ordinary imports for this path | Met | `lxml` removed; blocked-`lxml` smoke passed |
| Prompts/parsing constrained; no arbitrary prompts | Met | Preset prompt unchanged; grounded-markdown contract enforced |
| Validation lists exact commands, failures, skips, unrun checks | Met | Validation evidence below |
| Final report addresses eng/maintainer/QA/PM/DevOps | Met | This document |

## Engineering handoff

### Changed files

| File | Change |
|---|---|
| `docling/utils/deepseekocr_utils.py` | Remove `lxml`; reuse Chandra stdlib table parser; diagnostics helper; broader `[[...]]` payload capture with coordinate validation |
| `docling/pipeline/vlm_pipeline.py` | Consume diagnostics; single DeepSeek format error; include stop-reason context when incomplete; `_determine_status` skips duplicate stop-reason error for those pages |
| `tests/test_deepseekocr_vlm.py` | Core CI suite (no CI marker): matrix, stop-reason dedup, nonnumeric coords, `lxml` smoke, Ollama model-id helpers |
| `tests/test_deepseekocr_vlm_external.py` | New `external_service` live Ollama test using `VlmConvertOptions.from_preset("deepseek_ocr")` |
| `docs/usage/vision_models.md` | Document DeepSeek failure semantics |
| `docs/usage/model_catalog.md` | Note grounded parse failures → `partial_success` / `INFERENCE_FAILURE` |
| `.plans/completed/deepseek-ocr.md` | Approved plan + final handoff disposition (moved from active) |
| `demo/deepseek-ocr/change-report.md` | This handoff report |

### Architecture decisions

- DeepSeek-OCR is a multi-usage family; first slice hardens the **VLM adapter**
  only (API Ollama/LM Studio + grounded markdown).
- No local inference backbone yet (would be dead scaffolding).
- Public `parse_deepseekocr_markdown(...) -> DoclingDocument` retained; pipeline
  uses internal diagnostics.
- Classic OCR / post-processing remain follow-ups under `stages/ocr/` etc.

### Backward-compatibility impact

- Happy-path grounded-markdown fixtures unchanged (stdlib table parser parity).
- Unknown labels remain `TEXT`.
- Blank-but-present predictions remain empty success pages.
- Intentional change: nonblank zero-surviving-annotation responses become
  `PARTIAL_SUCCESS` + one `INFERENCE_FAILURE` (including stop-reason context
  when `LENGTH` / `CONTENT_FILTERED`).

### Dependency or lockfile impact

- No `pyproject.toml` / `uv.lock` changes.
- Removed undeclared `lxml` dependency from the DeepSeek parser path.

### Follow-up work

- Live Ollama E2E with `deepseek-ocr:3b` when service/model available.
- Classic OCR adapter and/or local inference only after new plan approval.
- Optional: lazy-load legacy `VlmPipeline` Transformers imports for slim API-only installs.
- Optional: stricter geometry validation (inverted / out-of-range boxes).

## Validation evidence

| Command | Environment | Result | Notes |
|---|---|---|---|
| `git diff --check` | local workspace | **Passed** | exit 0 |
| `uv run pytest tests/test_deepseekocr_vlm.py -v` | local after `make setup` | **Passed** | 16 passed |
| `uv run pytest tests/test_deepseekocr_vlm_external.py -v -rs` | local | **Skipped** | `Ollama is not available` |
| `uv run python .github/scripts/pytest_marker_selection.py core-ignore-args` | local | **Passed** | external module ignored; core DeepSeek module not ignored |
| `make validate` | local | **Passed** (rerun) | first run failed only after Ruff auto-format mutation |
| `make check` | local | **Failed** | pre-existing unrelated max-lines failures (below) |
| CUDA / Transformers local DeepSeek | n/a | **Not run** | out of approved scope |
| MPS | n/a | **Not run** | out of approved scope |

### Pre-existing `make check` failures

Unrelated to this slice; files were not modified by the contribution:

- `docling/backend/mspowerpoint_backend.py` (1421 lines)
- `docling/backend/xml/jats_backend.py` (1174 lines)
- `docling/datamodel/document.py` (1131 lines)

Ruff format/lint and Tach checks for the DeepSeek changeset were clean under
`make validate`.

### Skipped / unverified

- Live conversion with Ollama + `deepseek-ocr:3b` (service unavailable).
- Actual Ollama model-presence skip path against a live `/v1/models` listing
  (helper covered unit-wise; live branch not exercised).
- Claims from `demo/deepseek-ocr/maintainer-context.md` about PR #2721 coverage,
  empty CLI output, `infer()` / KV-cache failures, and the manual CUDA sample
  were **not** independently reproduced.
- Blank pages vs empty model output remain indistinguishable by design.

Do not treat skips as passes.

## QA handoff

- Scenarios covered:
  - Happy-path fixture GT parity
  - Unknown label → TEXT
  - Malformed arity + nonnumeric coordinates (mixed and only)
  - Nonblank unparseable → single format failure
  - Blank prediction → no DeepSeek error
  - Missing prediction → only `_determine_status` error
  - Unparseable + `LENGTH` / `CONTENT_FILTERED` → exactly one combined error
  - Parseable + `LENGTH` → existing stop-reason error retained
  - `lxml` blocked import/parse smoke
  - Preset construction + Ollama model-id helpers
- Edge cases: stop-reason dedup; nonnumeric-only format failure; external skip
  reason split (service vs model)
- Regression risks: callers that relied on silent empty docs for nonblank
  garbage now see `PARTIAL_SUCCESS`
- Hardware-dependent gaps: no live Ollama/LM Studio/CUDA/MPS execution

## PM handoff

- Problem addressed: Safer, reviewable DeepSeek-OCR VLM integration without
  repeating PR #2721 late discoveries.
- Delivered scope: Reliable API-only grounded-markdown conversion contract,
  docs, and core CI tests.
- Deferred scope and rationale: classic OCR, local inference, MPS — need
  proven contracts and would expand beyond the approved vertical slice.
- Decision or investment needed next: optional live-service validation; decide
  whether a classic OCR adapter or local backbone is the next funded slice.

## DevOps handoff

- Supported runtime and device matrix for this slice: API-only (Ollama / LM
  Studio). No CUDA/MPS claim.
- Optional dependencies: none added; `lxml` no longer required for DeepSeek
  parsing.
- Model download, cache, and network: live path uses remote Ollama; core tests
  do not download models.
- CI requirements: pure tests in `tests/test_deepseekocr_vlm.py` run in core CI;
  `tests/test_deepseekocr_vlm_external.py` is `external_service` and ignored by
  core selection.
- Deployment or observability: new/changed conversion errors are structured
  `ErrorItem`s with stable DeepSeek format message prefix; incomplete stop
  reasons may appear in the same message.

## Maintainer review

### Convention and architecture findings

| Finding | Severity | Resolution |
|---|---|---|
| Undeclared `lxml` in reliability slice | High (plan) | Fixed: reuse Chandra stdlib table parser; smoke-tested |
| Test marker organization vs CI enforcer | High (plan) | Fixed: separate `external_service` module |
| Unknown-label warn+skip broke compatibility | High (plan) | Fixed in plan/impl: keep TEXT fallback |
| Duplicate missing-prediction / empty errors | High (plan) | Fixed: missing → `_determine_status` only |
| Parser API contract unresolved | Medium (plan) | Locked: public `DoclingDocument` + internal diagnostics |
| Underspecified malformed matrix | Medium (plan) | Locked matrix + tests |
| Empty-as-failure without evidence | Medium (plan) | Blank present stays success |
| Weak minimal-install smoke | Medium (plan) | `lxml`-blocked smoke + helpers |
| Optional docs despite user-visible change | Medium (plan) | Docs mandatory and updated |
| Duplicate parser + stop-reason errors | P2 (impl review) | Combined message; `_determine_status` skips second |
| Nonnumeric coords not captured by regex | P2 (impl review) | Broadened payload capture; validate in helper |
| External test used legacy `DEEPSEEKOCR_OLLAMA` | P2 (impl review) | Switched to `from_preset("deepseek_ocr")` |
| No Ollama model-id availability check | P2 (impl review) | Check + distinct skip reasons + unit helpers |
| Second impl review pass | — | No new findings; human accepted |

### Unresolved questions

- None blocking for this slice.
- Next-slice choice (classic OCR vs local VLM inference) remains open.

### Recommended disposition

**Accept as PR-ready** with explicit non-blocking gaps: live Ollama E2E skipped;
`make check` red solely due to pre-existing unrelated max-lines debt.

Suggested PR title: `fix: harden DeepSeek-OCR API VLM grounded-markdown path`

Suggested PR body basis: this report’s executive summary, changed files,
validation table, and deferred scope.
