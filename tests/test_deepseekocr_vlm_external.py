"""External-service DeepSeek OCR VLM conversion tests (Ollama)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from .test_deepseekocr_vlm import (
    ollama_hosts_required_model,
    ollama_model_ids_from_response,
    required_deepseek_ollama_model_id,
)

pytestmark = pytest.mark.external_service


def test_e2e_deepseekocr_conversion() -> None:
    """Test DeepSeek OCR VLM conversion on a PDF file via local Ollama."""
    if os.getenv("CI"):
        pytest.skip("Skipping in CI environment")

    vlm_options = VlmConvertOptions.from_preset("deepseek_ocr")
    assert vlm_options.engine_options.engine_type == VlmEngineType.API_OLLAMA
    required_model = required_deepseek_ollama_model_id(vlm_options)
    assert required_model == "deepseek-ocr:3b"

    try:
        import requests

        response = requests.get("http://localhost:11434/v1/models", timeout=2)
    except Exception:
        pytest.skip("Ollama is not available")

    if response.status_code != 200:
        pytest.skip("Ollama is not available")

    try:
        payload = response.json()
    except Exception:
        pytest.skip("Ollama is not available")

    model_ids = ollama_model_ids_from_response(payload)
    if not ollama_hosts_required_model(model_ids, required_model):
        pytest.skip(f"Required Ollama model unavailable: {required_model}")

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )
    assert pipeline_options.vlm_options.engine_options.engine_type == (
        VlmEngineType.API_OLLAMA
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    pdf_path = Path("./tests/data/pdf/sources/2206.01062.pdf")
    conv_result = converter.convert(pdf_path)

    ref_path = Path("./tests/data/md_deepseek/groundtruth/deepseek_title.md.json")
    ref_doc = DoclingDocument.load_from_json(ref_path)

    doc = conv_result.document

    assert len(doc.pages) == 9, f"Number of pages mismatch: {len(doc.pages)}"
    assert len(doc.texts) > 0, "Document should have text elements"
    assert len(doc.pictures) > 0, "Document should have picture elements"

    title_texts = [t for t in doc.texts if t.label == "title"]
    assert len(title_texts) > 0, "Document should have a title"

    section_headers = [t for t in doc.texts if t.label == "section_header"]
    assert len(section_headers) > 0, "Document should have section headers"

    ref_title_texts = [t for t in ref_doc.texts if t.label == "title"]
    assert len(title_texts) == len(ref_title_texts), (
        f"Title count mismatch: {len(title_texts)} vs {len(ref_title_texts)}"
    )
