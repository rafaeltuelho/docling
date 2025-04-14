import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Optional, Type, Union

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureItem,
)
from docling_core.types.doc.document import (  # TODO: move import to docling_core.types.doc
    PictureDescriptionData,
)
from PIL import Image

from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PictureDescriptionBaseOptions,
)
from docling.models.base_model import (
    BaseItemAndImageEnrichmentModel,
    BaseModelWithOptions,
    ItemAndImageEnrichmentElement,
)


class PictureDescriptionBaseModel(
    BaseItemAndImageEnrichmentModel, BaseModelWithOptions
):
    images_scale: float = 2.0

    def __init__(
        self,
        *,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionBaseOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options
        self.provenance = "not-implemented"

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def _get_surrounding_text(self, doc: DoclingDocument, picture_item: PictureItem) -> str:
        """Get text context from items before and after the picture."""
        context = []
        found_picture = False
        before_count = 0
        after_count = 0
        
        for item, _ in doc.iterate_items():
            if item == picture_item:
                found_picture = True
                continue
                
            if not found_picture:
                if isinstance(item, (str, NodeItem)) and before_count < self.options.text_context_window_size:
                    text = item if isinstance(item, str) else getattr(item, 'text', '')
                    if text and text.strip():
                        context.append(f"Before image: {text}")
                        before_count += 1
            else:
                if isinstance(item, (str, NodeItem)) and after_count < self.options.text_context_window_size:
                    text = item if isinstance(item, str) else getattr(item, 'text', '')
                    if text and text.strip():
                        context.append(f"After image: {text}")
                        after_count += 1
                        
            if after_count >= self.options.text_context_window_size:
                break
                
        return "\n".join(context)

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        raise NotImplementedError

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        images: List[Image.Image] = []
        elements: List[PictureItem] = []
        contexts: List[str] = []
        
        for el in element_batch:
            assert isinstance(el.item, PictureItem)
            elements.append(el.item)
            images.append(el.image)
            context = self._get_surrounding_text(doc, el.item)
            contexts.append(context)

        outputs = self._annotate_with_context(images, contexts)

        for item, output in zip(elements, outputs):
            item.annotations.append(
                PictureDescriptionData(text=output, provenance=self.provenance)
            )
            yield item

    def _annotate_with_context(self, images: Iterable[Image.Image], contexts: List[str]) -> Iterable[str]:
        """Override this method to support context in concrete implementations."""
        # Default implementation ignores context
        yield from self._annotate_images(images)

    @classmethod
    @abstractmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        pass
