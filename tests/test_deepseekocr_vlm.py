"""Hardware-independent DeepSeek OCR grounded-markdown parser and pipeline tests."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from docling_core.types.doc import DocItemLabel, DoclingDocument, Size
from PIL import Image as PILImage

from docling.datamodel.base_models import (
    ConversionStatus,
    FailureCategory,
    Page,
    PagePredictions,
    VlmPrediction,
    VlmStopReason,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.deepseekocr_utils import (
    DEEPSEEK_OCR_FORMAT_ERROR,
    _parse_deepseekocr_markdown_with_diagnostics,
    parse_deepseekocr_markdown,
)

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA

_MD_SOURCES = Path("./tests/data/md_deepseek/sources/")
_EXAMPLE_MD = _MD_SOURCES / "deepseek_example.md"


def required_deepseek_ollama_model_id(options: VlmConvertOptions) -> str:
    """Return the Ollama model id configured by the DeepSeek-OCR preset."""
    api_override = options.model_spec.api_overrides.get(VlmEngineType.API_OLLAMA)
    if api_override is not None:
        model = api_override.params.get("model")
        if isinstance(model, str) and model:
            return model
    return options.model_spec.default_repo_id


def ollama_model_ids_from_response(payload: dict[str, Any]) -> set[str]:
    """Extract model ids from an OpenAI-compatible ``/v1/models`` payload."""
    data = payload.get("data", [])
    if not isinstance(data, list):
        return set()
    model_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.add(model_id)
    return model_ids


def ollama_hosts_required_model(
    model_ids: Iterable[str], required_model_id: str
) -> bool:
    """Return whether the required Ollama model id is present."""
    return required_model_id in set(model_ids)


def get_md_deepseek_paths() -> list[Path]:
    """Get all DeepSeek markdown test files."""
    return sorted(_MD_SOURCES.glob("*.md"))


def mock_parsing(content: str, filename: str) -> DoclingDocument:
    """Parse DeepSeek OCR markdown with a mock page image."""
    page = Page(page_no=1)
    page._image_cache[1.0] = PILImage.new("RGB", (612, 792), color="white")
    page.predictions = PagePredictions()
    page.predictions.vlm_response = VlmPrediction(text=content)

    return parse_deepseekocr_markdown(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_image=page.image,
        page_no=1,
        filename=filename,
    )


def _make_pipeline() -> VlmPipeline:
    return VlmPipeline.__new__(VlmPipeline)


def _make_page(
    page_no: int,
    *,
    text: str | None,
    missing_prediction: bool = False,
    stop_reason: VlmStopReason = VlmStopReason.UNSPECIFIED,
) -> Page:
    page = Page(page_no=page_no)
    page.size = Size(width=612, height=792)
    page._image_cache[1.0] = PILImage.new("RGB", (612, 792), color="white")
    backend = MagicMock()
    backend.is_valid.return_value = True
    page._backend = backend
    if missing_prediction:
        page.predictions = PagePredictions(vlm_response=None)
    else:
        assert text is not None
        page.predictions = PagePredictions(
            vlm_response=VlmPrediction(text=text, stop_reason=stop_reason)
        )
    return page


def _make_conv_res(pages: list[Page]) -> ConversionResult:
    conv_res = MagicMock(spec=ConversionResult)
    conv_res.pages = pages
    conv_res.errors = []
    conv_res.status = ConversionStatus.STARTED
    conv_res.input = MagicMock()
    conv_res.input.file = MagicMock()
    conv_res.input.file.name = "sample.pdf"
    conv_res.input._backend = None
    return conv_res


def test_e2e_deepseekocr_parsing() -> None:
    """Test DeepSeek OCR markdown parsing for all test files."""
    for md_path in get_md_deepseek_paths():
        annotated_content = md_path.read_text(encoding="utf-8")
        gt_path = md_path.parent.parent / "groundtruth" / md_path.name
        doc = mock_parsing(annotated_content, md_path.name)

        pred_md = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        pred_itxt = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )


def test_unknown_label_falls_back_to_text() -> None:
    content = "weird_label[[10, 20, 110, 40]]\nRetained unknown label content\n"
    diagnostics = _parse_deepseekocr_markdown_with_diagnostics(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="unknown.md",
    )

    assert diagnostics.format_error is None
    assert any("Unknown DeepSeek OCR grounded label" in w for w in diagnostics.warnings)
    texts = [t for t in diagnostics.document.texts if t.label == DocItemLabel.TEXT]
    assert len(texts) == 1
    assert texts[0].text == "Retained unknown label content"


def test_malformed_coordinates_skipped_with_valid_sibling() -> None:
    content = (
        "text[[10, 20, 110, 40]]\nKeep me\n"
        "text[[1, 2, 3]]\nDrop me\n"
        "text[[50, 60, 150, 80]]\nAlso keep me\n"
    )
    diagnostics = _parse_deepseekocr_markdown_with_diagnostics(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="mixed.md",
    )

    assert diagnostics.format_error is None
    assert any("malformed coordinates" in w for w in diagnostics.warnings)
    texts = [t.text for t in diagnostics.document.texts]
    assert texts == ["Keep me", "Also keep me"]


def test_nonnumeric_coordinates_skipped_with_valid_sibling() -> None:
    content = "text[[10, 20, 110, 40]]\nKeep me\ntext[[a, b, c, d]]\nDrop me\n"
    diagnostics = _parse_deepseekocr_markdown_with_diagnostics(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="nonnumeric-mixed.md",
    )

    assert diagnostics.format_error is None
    assert any("non-numeric coordinates" in w for w in diagnostics.warnings)
    texts = [t.text for t in diagnostics.document.texts]
    assert texts == ["Keep me"]


def test_nonnumeric_only_output_sets_format_error() -> None:
    content = "text[[a, b, c, d]]\nNo surviving annotations\n"
    diagnostics = _parse_deepseekocr_markdown_with_diagnostics(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="nonnumeric-only.md",
    )

    assert diagnostics.format_error == DEEPSEEK_OCR_FORMAT_ERROR
    assert any("non-numeric coordinates" in w for w in diagnostics.warnings)
    assert len(diagnostics.document.texts) == 0


def test_nonblank_unparseable_sets_format_error() -> None:
    content = "Just plain markdown without grounded annotations.\n"
    diagnostics = _parse_deepseekocr_markdown_with_diagnostics(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="plain.md",
    )

    assert diagnostics.format_error == DEEPSEEK_OCR_FORMAT_ERROR
    assert len(diagnostics.document.texts) == 0


def test_blank_prediction_has_no_format_error() -> None:
    diagnostics = _parse_deepseekocr_markdown_with_diagnostics(
        content="   \n",
        original_page_size=Size(width=612, height=792),
        page_no=1,
        filename="blank.md",
    )

    assert diagnostics.format_error is None
    assert diagnostics.warnings == []
    assert len(diagnostics.document.texts) == 0


def test_pipeline_unparseable_emits_exactly_one_inference_failure() -> None:
    pipeline = _make_pipeline()
    conv_res = _make_conv_res(
        [_make_page(1, text="plain markdown without annotations")]
    )

    pipeline._parse_deepseekocr_markdown(conv_res)
    status = pipeline._determine_status(conv_res)

    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    error = conv_res.errors[0]
    assert error.category == FailureCategory.INFERENCE_FAILURE
    assert error.page_no == 1
    assert error.error_message == DEEPSEEK_OCR_FORMAT_ERROR


def test_pipeline_unparseable_with_length_stop_reason_is_single_error() -> None:
    pipeline = _make_pipeline()
    conv_res = _make_conv_res(
        [
            _make_page(
                1,
                text="plain markdown without annotations",
                stop_reason=VlmStopReason.LENGTH,
            )
        ]
    )

    pipeline._parse_deepseekocr_markdown(conv_res)
    status = pipeline._determine_status(conv_res)

    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    error = conv_res.errors[0]
    assert error.category == FailureCategory.INFERENCE_FAILURE
    assert error.page_no == 1
    assert error.error_message.startswith(DEEPSEEK_OCR_FORMAT_ERROR)
    assert "stop_reason=length" in error.error_message


def test_pipeline_unparseable_with_content_filtered_is_single_error() -> None:
    pipeline = _make_pipeline()
    conv_res = _make_conv_res(
        [
            _make_page(
                1,
                text="plain markdown without annotations",
                stop_reason=VlmStopReason.CONTENT_FILTERED,
            )
        ]
    )

    pipeline._parse_deepseekocr_markdown(conv_res)
    status = pipeline._determine_status(conv_res)

    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    error = conv_res.errors[0]
    assert error.category == FailureCategory.INFERENCE_FAILURE
    assert error.page_no == 1
    assert error.error_message.startswith(DEEPSEEK_OCR_FORMAT_ERROR)
    assert "stop_reason=content_filter" in error.error_message


def test_pipeline_parseable_with_length_keeps_stop_reason_error() -> None:
    pipeline = _make_pipeline()
    conv_res = _make_conv_res(
        [
            _make_page(
                1,
                text="text[[10, 20, 110, 40]]\nKeep me\n",
                stop_reason=VlmStopReason.LENGTH,
            )
        ]
    )

    pipeline._parse_deepseekocr_markdown(conv_res)
    status = pipeline._determine_status(conv_res)

    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 1
    assert "VLM output incomplete" in conv_res.errors[0].error_message
    assert "stop_reason=length" in conv_res.errors[0].error_message


def test_pipeline_blank_prediction_has_no_deepseek_error() -> None:
    pipeline = _make_pipeline()
    conv_res = _make_conv_res([_make_page(1, text="   ")])

    pipeline._parse_deepseekocr_markdown(conv_res)
    status = pipeline._determine_status(conv_res)

    assert status == ConversionStatus.SUCCESS
    assert conv_res.errors == []


def test_pipeline_missing_prediction_not_duplicated_by_parser() -> None:
    pipeline = _make_pipeline()
    conv_res = _make_conv_res([_make_page(1, text=None, missing_prediction=True)])

    pipeline._parse_deepseekocr_markdown(conv_res)
    assert conv_res.errors == []

    status = pipeline._determine_status(conv_res)
    assert status == ConversionStatus.PARTIAL_SUCCESS
    assert len(conv_res.errors) == 1
    assert conv_res.errors[0].error_message == "No VLM prediction."
    assert conv_res.errors[0].category == FailureCategory.INFERENCE_FAILURE
    assert conv_res.errors[0].page_no == 1


def test_deepseek_ocr_preset_constructs() -> None:
    options = VlmConvertOptions.from_preset("deepseek_ocr")
    assert options.model_spec.name == "DeepSeek-OCR-3B"
    assert options.engine_options.engine_type == VlmEngineType.API_OLLAMA
    assert required_deepseek_ollama_model_id(options) == "deepseek-ocr:3b"
    pipeline_options = VlmPipelineOptions(vlm_options=options)
    assert pipeline_options.vlm_options.model_spec.response_format.value == (
        "deepseekocr_markdown"
    )


def test_ollama_model_id_helpers() -> None:
    options = VlmConvertOptions.from_preset("deepseek_ocr")
    required = required_deepseek_ollama_model_id(options)
    assert required == "deepseek-ocr:3b"

    present = ollama_model_ids_from_response(
        {"object": "list", "data": [{"id": "deepseek-ocr:3b"}, {"id": "llama3.2"}]}
    )
    assert present == {"deepseek-ocr:3b", "llama3.2"}
    assert ollama_hosts_required_model(present, required) is True

    absent = ollama_model_ids_from_response(
        {"object": "list", "data": [{"id": "llama3.2:latest"}]}
    )
    assert ollama_hosts_required_model(absent, required) is False
    assert ollama_model_ids_from_response({"data": "bad"}) == set()


def test_parse_succeeds_when_lxml_is_blocked() -> None:
    """DeepSeek parser must not require undeclared lxml at import/parse time."""
    assert _EXAMPLE_MD.is_file()
    script = f"""
import sys
sys.modules['lxml'] = None
from pathlib import Path
from docling_core.types.doc import Size
from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown
content = Path({str(_EXAMPLE_MD)!r}).read_text(encoding='utf-8')
doc = parse_deepseekocr_markdown(content, Size(width=612, height=792), page_no=1)
assert len(doc.tables) >= 1, 'expected table from fixture'
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
