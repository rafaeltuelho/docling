# Maintainer context from PR #2721

Source: <https://github.com/docling-project/docling/pull/2721>

This file preserves review knowledge from the closed, unmerged PR so a new
contributor can discover it before implementation. It is historical evidence;
current Docling code and current maintainer direction take precedence.

## Recovered identifiers

- Title: `feat: Add DeepSeek-OCR integration`
- Base commit: `e413e688ed5d4ce68c54cf47a1e6e3a028883c04`
- Last PR head commit: `234201c78a3cb4bea23a287bce617535c1f037ec`
- Original head branch: `feature/deepseek-ocr-integration`
- State: closed, unmerged

## Files changed by the original PR

- `docling/datamodel/pipeline_options.py`
- `docling/models/deepseek_ocr_model.py`
- `docling/models/plugins/defaults.py`
- `docs/getting_started/installation.md`
- `pyproject.toml`
- `tests/test_deepseek_ocr_model.py`
- `tests/test_e2e_ocr_conversion.py`
- `uv.lock`

Several of these paths or abstractions have since changed. Do not replay this
file list as an implementation plan.

## Architectural feedback to move earlier

- First decide whether DeepSeek-OCR is an OCR engine, a VLM pipeline model, or a
  reusable backbone supporting more than one usage.
- Separate the model backbone from usage-specific adapters.
- Account for three intended usages: VLM document conversion, classic OCR, and
  future post-processing OCR.
- A classic OCR adapter must constrain the prompt and parse the text-line output
  Docling can consume. The model may sometimes return markdown or grounded
  labels instead, so failure behavior must be explicit.
- MPS was paused because the community model and recent Transformers versions
  showed incompatible cache and operator behavior.
- Scale and image-processing defaults require model-specific evidence; copying
  values from EasyOCR or Tesseract is not sufficient.
- Optional dependencies should include only what runtime paths actually need.
- Hardware-dependent tests should detect compatible CUDA or MPS capability and
  skip clearly instead of relying only on an environment-variable gate.

## Observed quality and runtime signals

- Patch coverage was about 60.6%, with most uncovered lines in the model wrapper.
- Maintainer CLI testing produced an empty output for one test document.
- Community testing reported incompatible `infer()` arguments and Transformers
  KV-cache API failures.
- The PR accumulated conflicts as the repository evolved.
- One manual CUDA test succeeded on a sample, but that did not establish broad
  compatibility or CI coverage.

## Ground-truth recovery

GitHub normally retains the head of a closed PR under its pull ref. From a clean
checkout, recover it without merging it into the demo branch:

```bash
git fetch https://github.com/docling-project/docling.git \
  pull/2721/head:pr-2721-ground-truth
git show --stat pr-2721-ground-truth
git diff e413e688ed5d4ce68c54cf47a1e6e3a028883c04..pr-2721-ground-truth
```

Use that branch for comparison only. Port behavior deliberately onto current
`main`; do not cherry-pick the original implementation wholesale.
