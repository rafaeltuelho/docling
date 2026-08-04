"""Utilities for parsing DeepSeek OCR annotated markdown format."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ImageRef,
    ProvenanceItem,
    RefItem,
    Size,
    TextItem,
)
from PIL import Image as PILImage

from docling.utils.chandra_utils import _parse_table_html

_log = logging.getLogger(__name__)

DEEPSEEK_OCR_FORMAT_ERROR = (
    "DeepSeek OCR grounded-markdown parse failure: no valid annotations retained."
)

_LABEL_MAP: dict[str, DocItemLabel] = {
    "text": DocItemLabel.TEXT,
    "title": DocItemLabel.TITLE,
    "sub_title": DocItemLabel.SECTION_HEADER,
    "table": DocItemLabel.TABLE,
    "table_caption": DocItemLabel.CAPTION,
    "figure": DocItemLabel.PICTURE,
    "figure_caption": DocItemLabel.CAPTION,
    "image": DocItemLabel.PICTURE,
    "image_caption": DocItemLabel.CAPTION,
    "header": DocItemLabel.PAGE_HEADER,
    "footer": DocItemLabel.PAGE_FOOTER,
}

# Pattern to match: <|ref|>label<|/ref|><|det|>[[...]]<|/det|> or label[[...]].
# Capture the full payload; arity/numeric validation happens in
# `_parse_coordinate_payload`.
_ANNOTATION_PATTERN = (
    r"^(?:<\|ref\|>)?(\w+)(?:<\|/ref\|>)?(?:<\|det\|>)?\[\[([^\]]*)\]\]"
    r"(?:<\|/det\|>)?\s*$"
)


@dataclass
class DeepSeekOcrParseDiagnostics:
    """Internal parse result consumed by VlmPipeline."""

    document: DoclingDocument
    warnings: list[str] = field(default_factory=list)
    format_error: str | None = None


def _collect_annotation_content(
    lines: list[str],
    i: int,
    label_str: str,
    annotation_pattern: str,
    visited_lines: set[int],
) -> tuple[str, int]:
    """Collect content for an annotation.

    Args:
        lines: All lines from the document
        i: Current line index (after annotation line)
        label_str: The annotation label (e.g., 'table', 'text')
        annotation_pattern: Regex pattern to match annotations
        visited_lines: Set of already visited line indices

    Returns:
        Tuple of (content string, next line index)
    """
    content_lines = []

    # Special handling for table: extract only <table>...</table>
    if label_str == "table":
        table_started = False
        ii = i
        while ii < len(lines):
            line = lines[ii]
            if "<table" in line.lower():
                table_started = True
            if table_started:
                visited_lines.add(ii)
                content_lines.append(line.rstrip())
            if table_started and "</table>" in line.lower():
                break
            ii += 1
    else:
        # Original logic for other labels
        while i < len(lines):
            content_line = lines[i].strip()
            if content_line:
                if re.match(annotation_pattern, content_line):
                    break
                visited_lines.add(i)
                content_lines.append(lines[i].rstrip())
                i += 1
                if label_str not in ["figure", "image"]:
                    break
            else:
                i += 1
                if content_lines:
                    break

    return "\n".join(content_lines), i


def _process_annotation_item(
    label_str: str,
    content: str,
    prov: ProvenanceItem,
    caption_item: Optional[Union[TextItem, RefItem]],
    page_doc: DoclingDocument,
    label_map: dict[str, DocItemLabel],
) -> None:
    """Process and add a single annotation item to the document.

    Args:
        label_str: The annotation label
        content: The content text
        prov: Provenance information
        caption_item: Optional caption item to link
        page_doc: Document to add item to
        label_map: Mapping of label strings to DocItemLabel
    """
    doc_label = label_map.get(label_str, DocItemLabel.TEXT)

    if label_str in ["figure", "image"]:
        page_doc.add_picture(caption=caption_item, prov=prov)
    elif label_str == "table":
        table_data = _parse_table_html(content)
        page_doc.add_table(data=table_data, caption=caption_item, prov=prov)
    elif label_str == "title":
        clean_content = content
        if content.startswith("#"):
            hash_count = 0
            for char in content:
                if char == "#":
                    hash_count += 1
                else:
                    break
            clean_content = content[hash_count:].strip()
        page_doc.add_title(text=clean_content, prov=prov)
    elif label_str == "sub_title":
        heading_level = 1
        clean_content = content
        if content.startswith("#"):
            hash_count = 0
            for char in content:
                if char == "#":
                    hash_count += 1
                else:
                    break
            if hash_count > 1:
                heading_level = hash_count - 1
            clean_content = content[hash_count:].strip()
        page_doc.add_heading(text=clean_content, level=heading_level, prov=prov)
    else:
        page_doc.add_text(label=doc_label, text=content, prov=prov)


def _parse_coordinate_payload(
    coords_str: str,
) -> tuple[list[float] | None, str | None]:
    """Parse an annotation coordinate payload into four floats.

    Returns:
        (coords, warning). coords is None when the payload is malformed.
    """
    try:
        coords = [float(x.strip()) for x in coords_str.split(",")]
    except ValueError:
        return None, f"Skipping annotation with non-numeric coordinates: {coords_str!r}"

    if len(coords) != 4:
        return (
            None,
            f"Skipping annotation with malformed coordinates "
            f"(expected 4 values, got {len(coords)}): {coords_str!r}",
        )
    return coords, None


def _parse_deepseekocr_markdown_with_diagnostics(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: Optional[PILImage.Image] = None,
) -> DeepSeekOcrParseDiagnostics:
    """Parse DeepSeek OCR markdown and return document plus diagnostics."""
    warnings: list[str] = []
    label_map = _LABEL_MAP

    origin = DocumentOrigin(
        filename=filename,
        mimetype="text/markdown",
        binary_hash=0,
    )
    page_doc = DoclingDocument(name=filename.rsplit(".", 1)[0], origin=origin)

    pg_width = original_page_size.width
    pg_height = original_page_size.height
    scale_x = pg_width / 1000
    scale_y = pg_height / 1000

    image_dpi = 72
    if page_image is not None:
        image_dpi = int(72 * page_image.width / pg_width)

    page_doc.add_page(
        page_no=page_no,
        size=Size(width=pg_width, height=pg_height),
        image=ImageRef.from_pil(image=page_image, dpi=image_dpi)
        if page_image
        else None,
    )

    lines = content.split("\n")
    annotations: list[tuple[str, str, ProvenanceItem]] = []
    i = 0
    visited_lines: set[int] = set()

    while i < len(lines):
        if i in visited_lines:
            i += 1
            continue

        line = lines[i].strip()
        match = re.match(_ANNOTATION_PATTERN, line)
        if match:
            label_str = match.group(1)
            coords_str = match.group(2)
            coords, coord_warning = _parse_coordinate_payload(coords_str)
            if coords is None:
                if coord_warning is not None:
                    warnings.append(coord_warning)
                    _log.warning(coord_warning)
                i += 1
                continue

            if label_str not in label_map:
                warning = (
                    f"Unknown DeepSeek OCR grounded label {label_str!r}; "
                    "falling back to TEXT."
                )
                warnings.append(warning)
                _log.warning(warning)

            bbox = BoundingBox(
                l=coords[0] * scale_x,
                t=coords[1] * scale_y,
                r=coords[2] * scale_x,
                b=coords[3] * scale_y,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            prov = ProvenanceItem(page_no=page_no, bbox=bbox, charspan=[0, 0])

            i += 1
            content_text, i = _collect_annotation_content(
                lines, i, label_str, _ANNOTATION_PATTERN, visited_lines
            )
            annotations.append((label_str, content_text, prov))
            continue
        i += 1

    for idx, (label_str, content_text, prov) in enumerate(annotations):
        caption_item = None
        if label_str in ["table", "figure", "image"] and idx + 1 < len(annotations):
            next_label, next_content, next_prov = annotations[idx + 1]
            if (
                (label_str == "table" and next_label == "table_caption")
                or (label_str == "figure" and next_label == "figure_caption")
                or (label_str == "image" and next_label == "image_caption")
            ):
                caption_label = label_map.get(next_label, DocItemLabel.CAPTION)
                caption_item = page_doc.add_text(
                    label=caption_label,
                    text=next_content,
                    prov=next_prov,
                )

        if label_str in ["figure_caption", "table_caption", "image_caption"]:
            if idx > 0:
                prev_label = annotations[idx - 1][0]
                if (
                    (label_str == "table_caption" and prev_label == "table")
                    or (label_str == "figure_caption" and prev_label == "figure")
                    or (label_str == "image_caption" and prev_label == "image")
                ):
                    continue

        _process_annotation_item(
            label_str, content_text, prov, caption_item, page_doc, label_map
        )

    format_error: str | None = None
    if content.strip() and len(annotations) == 0:
        format_error = DEEPSEEK_OCR_FORMAT_ERROR
        warnings.append(format_error)
        _log.warning(format_error)

    return DeepSeekOcrParseDiagnostics(
        document=page_doc,
        warnings=warnings,
        format_error=format_error,
    )


def parse_deepseekocr_markdown(
    content: str,
    original_page_size: Size,
    page_no: int,
    filename: str = "file",
    page_image: Optional[PILImage.Image] = None,
) -> DoclingDocument:
    """Parse DeepSeek OCR markdown with label[[x1, y1, x2, y2]] format.

    This function parses markdown content that has been annotated with bounding box
    coordinates for different document elements.

    Labels supported:
    - text: Standard body text
    - title: Main document or section titles
    - sub_title: Secondary headings or sub-headers
    - table: Tabular data
    - table_caption: Descriptive text for tables
    - figure: Image-based elements or diagrams
    - figure_caption: Titles or descriptions for figures/images
    - header / footer: Content at top or bottom margins of pages

    Args:
        content: The annotated markdown content string
        original_page_size: Physical page dimensions (points).
        page_no: Page number (1-based).
        filename: Source filename.
        page_image: Optional PIL image of the page.

    Returns:
        DoclingDocument with parsed content
    """
    return _parse_deepseekocr_markdown_with_diagnostics(
        content=content,
        original_page_size=original_page_size,
        page_no=page_no,
        filename=filename,
        page_image=page_image,
    ).document
