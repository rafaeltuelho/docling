# DeepSeek-OCR implementation plan

**Status:** Approved

Allowed values are `Proposed`, `Approved`, or `Superseded`. The agent must not
mark its own plan `Approved`.

## Repository snapshot

- Branch: `agent/docling-contribution-workflow`
- Base commit: `9b454c9e88454d95fd04d538c552a3c07bc3c04d` (`main`); HEAD `a5e4a4da2406908c24afc15707fda842660aabf6`
- Working tree state: plan-only edits under `.plans/active/deepseek-ocr.md`
- Date: 2026-08-04 (revised after independent verification)

## Requirements understood

- User or integration outcome: Make the existing API-only DeepSeek-OCR VLM
conversion path (`--pipeline vlm --vlm-model deepseek_ocr`) reliable and
reviewable by enforcing a constrained grounded-markdown contract, explicit
parser-format failure behavior for nonblank unparseable output, removal of the
undeclared `lxml` dependency from the parser path, and hardware-independent
tests that do not require GPU, model download, or remote-code execution.
- Acceptance criteria covered:
  - Compared current Nemotron classic OCR and Granite-Docling / DeepSeek VLM
  paths before choosing a seam.
  - Architecture treats DeepSeek as a multi-usage model family; this slice
  hardens only the VLM adapter and records why no local backbone is added yet.
  - First slice is user-visible reliability for an already-exposed preset, not
  dead scaffolding.
  - MPS remains out of scope.
  - Policy/parser tests run without GPU, network, remote code, or downloads.
  - Live Ollama/LM Studio tests live in a separate `external_service` module.
  - Ordinary Docling imports and the DeepSeek parser path do not require `lxml`
  or a new optional dependency extra.
  - Prompts stay fixed on the preset; arbitrary prompts are not exposed.
- Non-goals:
  - Classic OCR (`BaseOcrModel` / `TextCell`) adapter.
  - Local Transformers / vLLM / MLX inference.
  - Fixing legacy `VlmPipeline` module-level Transformers imports.
  - Post-processing OCR usage.
  - Apple Silicon / MPS support.
  - GLM-OCR or other model families.
  - Live multi-gigabyte model download during validation.
  - Replaying historical PR #2721 paths or file layout.
  - Treating blank-but-present predictions as failures (blank pages cannot be
  distinguished from empty model output without additional signals).
- Resolved decisions (locked for approval; supersede prior open questions 2–3):
  1. **Slice:** VLM API hardening only (not classic OCR / local backbone).
  2. **Blank prediction:** present `VlmPrediction` with empty/whitespace text
    keeps current success-with-empty-page behavior. Rationale: no
     model-specific evidence that empty text is always inference failure; blank
     source pages are indistinguishable. Incomplete generation remains covered
     by existing `_determine_status` stop-reason handling (`LENGTH`,
     `CONTENT_FILTERED`).
  3. **Unknown labels:** preserve current `DocItemLabel.TEXT` fallback (Chandra /
    Dots precedent). Optional warning log allowed. Skipping is reserved for
     structurally invalid annotations only.
  4. **Parser API:** keep public
    `parse_deepseekocr_markdown(...) -> DoclingDocument`. Add a typed internal
     diagnostic result consumed by `VlmPipeline`; do not change the public
     return type and do not make the public function raise for policy failures.
  5. **Missing prediction:** handled only by `_determine_status`; the DeepSeek
    parser path must not emit a second error for `vlm_response is None`.
- Remaining open question:
  1. Confirm the first slice remains VLM API hardening as specified here. If
    maintainers want classic OCR or local inference instead, supersede this
     plan rather than expanding the boundary mid-implementation.



## Repository evidence


| Area                      | Current source or test                                                                                                                                                                                                                                                        | Relevant convention                                                                                                                                                                                           | Consequence                                                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| OCR integration           | `docling/models/stages/ocr/nemotron_ocr_model.py`, `BaseOcrModel`, pluggy `defaults.ocr_engines()`                                                                                                                                                                            | Classic OCR returns positioned `TextCell`s; optional deps lazy; CUDA validated in production `validate_runtime()`; tests capability-skip via that validator                                                   | A DeepSeek classic OCR adapter would live under `stages/ocr/`, constrain prompt/grammar, and not reuse the grounded-markdown document parser |
| VLM or inference backbone | `stages/vlm_convert/vlm_convert_model.py`, `inference_engines/vlm/`, `GRANITE_DOCLING_MODEL_SPEC_BASE`, existing `VLM_CONVERT_DEEPSEEK_OCR`                                                                                                                                   | New VLM work uses stage presets + engines; Granite shares a model-spec base across usages; DeepSeek is already API-only (`API_OLLAMA`, `API_LMSTUDIO`) with fixed grounding prompt and `DEEPSEEKOCR_MARKDOWN` | First slice stays on the VLM adapter/parser; do not invent a local backbone until Transformers/vLLM contracts are proven                     |
| Options and public API    | `stage_model_specs.py` (`deepseek_ocr`), legacy `vlm_model_specs.DEEPSEEKOCR_OLLAMA`, CLI `--vlm-model`                                                                                                                                                                       | Preset IDs come from `VlmConvertOptions.list_preset_ids()`; OCR engines from factory enum                                                                                                                     | No new CLI flag needed for this slice; keep prompt fixed on the preset                                                                       |
| Plugin registration       | OCR: pluggy entry point `docling` → `defaults.ocr_engines()`; VLM: in-process `register_preset`                                                                                                                                                                               | Historical flat `docling/models/deepseek_ocr_model.py` is obsolete                                                                                                                                            | Do not recreate the PR #2721 flat model wrapper                                                                                              |
| Dependencies              | DeepSeek parser imports `lxml` unconditionally; `lxml` is declared only under `format-xml-jats` in `pyproject.toml`; Chandra/Dots use stdlib `_TableHTMLParser` via `chandra_utils._parse_table_html`                                                                         | Optional heavy deps must stay out of ordinary import paths; Dots already reuses Chandra’s stdlib table parser                                                                                                 | This slice **must** remove DeepSeek’s `lxml` table parser and reuse the stdlib implementation; no new dependency extra / lockfile churn      |
| Tests and CI              | `tests/test_deepseekocr_vlm.py` (`pytestmark = ml_vlm`); `.github/scripts/pytest_marker_selection.py` requires CI markers only via module-level `pytestmark`; `external_service` modules are ignored by core CI; `tests/test_cli_remote.py` is the external-service precedent | Pure parser tests must run in core CI; live Ollama must be a separate module with module-level `external_service`                                                                                             | Split files; never put function-level CI markers in the pure module                                                                          |




### Obsolete vs current paths


| Historical PR #2721                    | Current equivalent                                                                     |
| -------------------------------------- | -------------------------------------------------------------------------------------- |
| `docling/models/deepseek_ocr_model.py` | OCR → `docling/models/stages/ocr/`; VLM → preset + `stages/vlm_convert/` + parser util |
| Single wrapper for all usages          | Usage-specific adapters; shared local inference only when a second consumer needs it   |
| OCR enum extension                     | Deprecated; OCR plugin factory / `kind` on options                                     |
| Mixed unit + E2E in one opaque suite   | Separate policy/parser (core CI) and external-service modules                          |




## Architecture decision

- Proposed seam: Treat DeepSeek-OCR as a **reusable model family with
usage-specific adapters**. The only adapter in scope now is the existing
**VLM document-conversion adapter** (API engines + grounded-markdown parser).
Do **not** introduce a local inference backbone in this slice because the
current preset has no Transformers/MLX/vLLM support and a backbone without a
second consumer would be dead scaffolding.
- Alternatives considered:
  1. **Classic OCR adapter first** — rejected for this slice. Grounded markdown
    / document items are not `TextCell` lines; a classic adapter needs a
     separate constrained grammar, confidence policy, and likely local or remote
     inference not yet settled. Nemotron remains the OCR precedent for a later
     slice.
  2. **Local Transformers/vLLM backbone first** — rejected. Current preset
    excludes local engines; MPS is out of scope. Historical PR signals about
     `infer()` / KV-cache are maintainer-context only and were not independently
     reproduced here.
  3. **Extract shared** `DEEPSEEK_OCR_MODEL_SPEC_BASE` **with no second user** —
    rejected as scaffolding-only.
  4. **Declare** `lxml` **in base/API extras** — rejected in favor of removing the
    undeclared dependency by reusing the existing stdlib table parser
     (Chandra/Dots precedent), which better matches modular `docling-slim`
     installs.
- Why this is the smallest coherent slice: DeepSeek VLM conversion is already
user-reachable via CLI/SDK, but undeclared `lxml`, silent unparseable
nonblank output, and mis-marked tests undermine reliability. Hardening that
path delivers a real outcome without new deps, devices, or unused modules.
- How it can extend to the other DeepSeek-OCR usages:
  - **Local VLM inference:** add engine support on a shared model-spec base only
  after a compatible checkpoint/API is evidenced; keep prompt/response_format
  on the VLM preset.
  - **Classic OCR:** new `stages/ocr/deepseek_ocr_model.py` + options `kind`,
  fixed text-line prompt/parser into `TextCell`, plugin registration; share
  local inference only if the VLM path also needs it.
  - **Post-processing:** separate adapter over page/OCR context; not designed here.
- Compatibility and migration impact:
  - Happy-path grounded-markdown fixtures and unknown-label TEXT fallback remain
  behavior-compatible.
  - Intentional user-visible change: nonblank responses with zero retained
  annotations become a parser-format `ErrorItem`
  (`FailureCategory.INFERENCE_FAILURE`) and contribute to `PARTIAL_SUCCESS`.
  - Blank-but-present predictions remain empty successful pages (compatibility
  preserved; documented limitation).
  - No public options schema change. Legacy `DEEPSEEKOCR_OLLAMA` left untouched.
  - Removing `lxml` from the DeepSeek parser is a dependency fix, not a
  document-shape change; table fixtures must keep equivalent structure via the
  stdlib parser.



### Explicit rationale for not coupling a backbone yet

The task brief requires not coupling a reusable backbone to a single usage
without rationale. Rationale: the supported runtime today is API-only; there is
no shared local loader to extract. Coupling a speculative backbone to this VLM
slice would invent an unused abstraction. When a second usage or local engine is
approved, follow the Granite `*_MODEL_SPEC_BASE` / inference-engine pattern and
keep stage adapters thin.

### Parser API design (locked)

```text
parse_deepseekocr_markdown(...) -> DoclingDocument
  public, stable; may wrap the internal helper and return only the document

_parse_deepseekocr_markdown_with_diagnostics(...) -> DeepSeekOcrParseDiagnostics
  internal TypedDict/dataclass used by VlmPipeline
  fields (names flexible, shape fixed):
    document: DoclingDocument
    warnings: list[str]          # malformed skips; optional unknown-label notes
    format_error: str | None     # set iff nonblank input and zero retained annotations
```

`VlmPipeline._parse_deepseekocr_markdown` must:

1. If `page.predictions.vlm_response is None`: build/skip with empty text **without**
  appending a DeepSeek-specific error (existing `_determine_status` owns
   “No VLM prediction.”).
2. If prediction present: call the diagnostics helper.
3. If `format_error` is set: append exactly one `ErrorItem` for that page
  (`component_type=PIPELINE`, `category=INFERENCE_FAILURE`, stable message
   naming DeepSeek OCR grounded-markdown parse failure) and set
   `PARTIAL_SUCCESS`.
4. Never raise from the public parser for these policy outcomes.



### Input / outcome matrix (locked)

Definitions used below:

- **Blank prediction:** `vlm_response` present and `text.strip() == ""`.
- **Malformed coordinates:** annotation line matches the label/det pattern but
the coordinate payload is not exactly four numeric values parseable as
`float` (wrong arity, nonnumeric tokens). These annotations are skipped.
- **Unknown label:** recognized annotation syntax with a label absent from the
DeepSeek label map → retain as `TEXT` (current behavior).
- **Unparseable page:** prediction text is nonblank and zero annotations are
retained after parsing (plain Markdown, garbage, or only malformed
annotations).
- Out-of-range coordinates (outside 0–1000), inverted boxes, and empty
annotation bodies are **not** newly rejected in this slice; current geometry /
content retention stays to avoid compatibility churn. Follow-up only if
maintainers later want stricter geometry validation.


| Input                                                        | Retained content                                          | Warnings                                                            | Errors this page                                               | Status effect                             |
| ------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------- |
| `vlm_response is None`                                       | empty page from parser path if invoked; no DeepSeek error | 0                                                                   | exactly 1 from `_determine_status` only (“No VLM prediction.”) | `PARTIAL_SUCCESS`                         |
| Blank prediction                                             | empty page document                                       | 0                                                                   | 0 from DeepSeek parser                                         | no status downgrade from DeepSeek parsing |
| Nonblank, ≥1 valid annotation                                | those items                                               | optional: unknown-label note; warn per malformed skipped annotation | 0                                                              | success from DeepSeek parsing             |
| Nonblank, mixed valid + malformed                            | valid items only                                          | warn per malformed skip                                             | 0                                                              | success from DeepSeek parsing             |
| Nonblank, unknown label + valid bbox/content                 | `TEXT` item                                               | optional warn                                                       | 0                                                              | success from DeepSeek parsing             |
| Nonblank, only malformed / no annotations retained           | empty page document                                       | warn as applicable                                                  | exactly 1 DeepSeek format `INFERENCE_FAILURE`                  | `PARTIAL_SUCCESS`                         |
| Existing happy-path fixtures under `tests/data/md_deepseek/` | unchanged GT                                              | 0 required                                                          | 0                                                              | success                                   |


Tests must assert **exactly one** error for each single-page failure case, plus
exact `page_no`, `category`, and message stability for the DeepSeek format
error.

## Approved change boundary


| File                                     | Change                                                                                                                                                                                                                                                                                                                               | Evidence or precedent                                                                                     | Validation                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `docling/utils/deepseekocr_utils.py`     | Remove `lxml` import and local `_parse_table_html`; reuse `docling.utils.chandra_utils._parse_table_html` (Dots precedent). Keep public `parse_deepseekocr_markdown(...) -> DoclingDocument`. Add internal diagnostics helper implementing the locked matrix. Preserve unknown-label TEXT fallback; skip only malformed coordinates. | `dots_utils.py` import; Chandra stdlib parser; current label fallback at `deepseekocr_utils.py`           | Unit/fixture tests; `lxml`-blocked smoke                                  |
| `docling/pipeline/vlm_pipeline.py`       | Consume diagnostics; emit exactly one DeepSeek format error for unparseable nonblank pages; do not double-report missing predictions                                                                                                                                                                                                 | DocLang fragment errors; `_determine_status` missing-prediction path                                      | Pipeline unit tests with mocked `VlmPrediction`                           |
| `tests/test_deepseekocr_vlm.py`          | Hardware-independent parser/policy/pipeline tests only; **no** module-level CI marker (`ml_vlm` / `external_service`). Cover matrix rows above and `lxml`-blocked import/parse smoke patterned on `tests/test_backend_optional_dependencies.py`                                                                                      | Marker script requires module-level markers; optional-deps smoke uses `sys.modules[name]=None` subprocess | `uv run pytest tests/test_deepseekocr_vlm.py` in core/default suite       |
| `tests/test_deepseekocr_vlm_external.py` | **New** module: move live Ollama conversion here with module-level `pytestmark = pytest.mark.external_service`, plus service/model availability skip reasons                                                                                                                                                                         | `tests/test_cli_remote.py`; core CI ignores `external_service` modules                                    | Not required for merge green in core CI; run manually when Ollama present |
| `tests/data/md_deepseek/`                | Add small fixtures only for matrix cases that need file-based GT (optional if in-test strings suffice)                                                                                                                                                                                                                               | Existing source/groundtruth layout                                                                        | Same core pytest module                                                   |
| `docs/usage/vision_models.md`            | **Mandatory:** document DeepSeek-OCR API-only grounded-markdown contract and failure semantics (unparseable nonblank → `PARTIAL_SUCCESS` + `INFERENCE_FAILURE`; blank prediction remains empty success; missing prediction unchanged)                                                                                                | AGENTS.md requires docs for user-facing behavior changes                                                  | Docs review in Validate                                                   |
| `docs/usage/model_catalog.md`            | **Mandatory:** one-line clarification that DeepSeek-OCR grounded-markdown parse failures surface as partial success / inference errors; still API-only                                                                                                                                                                               | Catalog already lists DeepSeek-OCR                                                                        | Docs review in Validate                                                   |


Files not listed here require a plan revision and new approval.

Out of boundary (follow-ups): `stages/ocr/deepseek_ocr_model.py`,
`pipeline_options.py` OCR options, `plugins/defaults.py`, `pyproject.toml` /
`uv.lock` (not needed once `lxml` is removed from the parser), extracting
Chandra’s table parser to a third shared module (reuse via existing import is
enough), Transformers/vLLM engines, lazy-loading legacy `VlmPipeline`
Transformers imports, MPS, model download tooling, e2e OCR GT suites, wholesale
legacy `DEEPSEEKOCR_OLLAMA` removal, stricter geometry validation for inverted /
out-of-range boxes.

## Implementation sequence

1. Replace DeepSeek’s `lxml` table parser with `chandra_utils._parse_table_html`;
  confirm happy-path table fixtures still match GT (or regenerate only if the
   stdlib parser intentionally differs—prefer behavioral parity).
2. Add internal diagnostics helper; keep public parser signature returning
  `DoclingDocument`.
3. Wire `VlmPipeline` to diagnostics with the locked missing-prediction /
  blank / unparseable split.
4. Restructure tests: pure coverage in `test_deepseekocr_vlm.py` (no CI marker);
  move live Ollama to `test_deepseekocr_vlm_external.py` with module-level
   `external_service`.
5. Add matrix assertions (exactly one error, category, page, message) and the
  `lxml`-blocked smoke.
6. Update `vision_models.md` and `model_catalog.md` with the documented failure
  semantics.
7. Stop for Validate/Review; do not expand into OCR or local inference without
  a new approval.



## Test matrix


| Scenario                                                                | Test level                                    | Hardware or dependency need                             | Expected result                                                                                                                                                    |
| ----------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Existing grounded-markdown fixtures parse to current GT                 | Unit / fixture in `test_deepseekocr_vlm.py`   | None                                                    | Pass; no GT churn unless stdlib table parity forces reviewed updates                                                                                               |
| Unknown label retained as TEXT                                          | Unit                                          | None                                                    | Item present; no format error; optional warning only                                                                                                               |
| Malformed coordinates skipped; sibling valid annotation kept            | Unit                                          | None                                                    | One retained item; 0 errors; warning for skip                                                                                                                      |
| Nonblank unparseable / zero retained annotations                        | Unit + pipeline mock                          | None                                                    | Exactly one `INFERENCE_FAILURE`; `PARTIAL_SUCCESS`; empty page doc                                                                                                 |
| Blank prediction                                                        | Unit + pipeline mock                          | None                                                    | 0 DeepSeek format errors; empty success page                                                                                                                       |
| `vlm_response is None`                                                  | Pipeline mock                                 | None                                                    | Exactly one error from `_determine_status`; no DeepSeek duplicate                                                                                                  |
| `lxml` blocked: import parser + parse fixture including a table         | Subprocess smoke (`sys.modules['lxml']=None`) | Dev env may have `lxml` installed; block forces absence | Exit code 0; parse succeeds                                                                                                                                        |
| Construct API-only options path without exercising Transformers engines | Unit                                          | None                                                    | `VlmConvertOptions.from_preset("deepseek_ocr")` succeeds; documents that full `VlmPipeline` import may still pull legacy Transformers (pre-existing, out of scope) |
| Live Ollama `deepseek_ocr` conversion                                   | `test_deepseekocr_vlm_external.py`            | Local Ollama + model                                    | Run only if available; else skip with reason; never call skip a pass                                                                                               |
| CUDA / Transformers local DeepSeek                                      | Not in slice                                  | N/A                                                     | Not run                                                                                                                                                            |
| MPS                                                                     | Not in slice                                  | N/A                                                     | Out of scope                                                                                                                                                       |




### Exact smoke commands (validation evidence)

Core / hardware-independent:

```bash
uv run pytest tests/test_deepseekocr_vlm.py -v
```

`lxml` absence smoke (also asserted inside the test module via subprocess helper
equivalent to `tests/test_backend_optional_dependencies.py`):

```bash
uv run python -c "import sys; sys.modules['lxml']=None; from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown; from docling_core.types.doc import Size; from pathlib import Path; content=Path('tests/data/md_deepseek/sources').glob('*.md').__iter__().__next__().read_text(); parse_deepseekocr_markdown(content, Size(width=612,height=792), page_no=1); print('ok')"
```

Acceptance: process exit code 0 and prints `ok`. Failure if `ImportError: lxml`
or equivalent occurs during import/parse.

External (optional; not a core CI gate):

```bash
uv run pytest tests/test_deepseekocr_vlm_external.py -v
```

Repository checks after implementation:

```bash
make validate
make check   # when time/environment allow
```



## Risks and limitations


| Risk                                                              | Impact                                                               | Mitigation or follow-up                                                                                                 |
| ----------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Stdlib table parser differs subtly from `lxml` HTML handling      | Table GT drift                                                       | Prefer parity; if GT must change, review diffs explicitly; keep scope to DeepSeek tables only                           |
| Blank pages vs failed empty model output remain indistinguishable | Empty success pages may hide some model failures                     | Document limitation; rely on stop-reason errors for truncated/filtered output; revisit only with model-specific signals |
| Legacy `VlmPipeline` module-level Transformers imports            | API-only installs may still need Transformers to import the pipeline | Pre-existing; out of slice; document in change report / DevOps notes                                                    |
| Maintainer may prefer classic OCR as the “real” contribution      | Plan may need superseding                                            | Remaining open question #1                                                                                              |
| Local Transformers/vLLM/MPS compatibility unknown                 | Local inference follow-up blocked                                    | Keep local engines and MPS out of scope until independently evidenced                                                   |
| Historical PR #2721 runtime quality claims                        | May over/under estimate local-inference difficulty                   | Treat maintainer-context as historical evidence only; do not cite as validation results                                 |




## Claims not independently verified in planning

These remain unsupported by commands run during planning and must not be
reported as validation passes later unless actually executed:

- PR #2721 coverage (~60.6%), empty-output CLI observation, `infer()` /
KV-cache failures, and the manual CUDA sample — sourced only from
`demo/deepseek-ocr/maintainer-context.md`.
- Actual Ollama/LM Studio output shape, model availability, identifiers, and
malformed-output frequency.
- Whether empty model outputs ever correspond to blank source pages in practice.
- Ecosystem availability of a compatible local Transformers/vLLM DeepSeek
checkpoint beyond the current API-only preset declaration.
- No GPU, model-download, remote-service, or end-to-end DeepSeek execution was
performed while authoring this plan.



## Human approval

- Approved by: Rafael Soares (maintainer proxy)

- Approved at: 2026-08-04

- Approved scope changes:

  - Confirmed VLM API hardening as the first implementation slice.

  - Approved only the files and behavior listed in the change boundary.

  - No additional scope was authorized.

- Notes:

  - Approved the locked parser API and input/outcome matrix.

  - Blank-but-present predictions retain the existing empty-page behavior.

  - Unknown labels remain preserved as TEXT.

  - Missing predictions are handled only by `_determine_status`.

  - Nonblank output with zero retained annotations produces exactly one

    INFERENCE_FAILURE.

  - Core parser, pipeline, dependency, and documentation validation is required.

  - Live Ollama validation is optional and must be reported as skipped if unavailable.

  - Legacy module-level Transformers imports and a full minimal-install

    VlmPipeline smoke test are accepted as documented, pre-existing limitations

    outside this slice.

  - Any scope expansion requires plan revision and renewed approval.

