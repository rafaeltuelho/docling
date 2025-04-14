from pathlib import Path
from typing import Iterable, Optional, Type, Union, List

from PIL import Image

from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionBaseOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.utils.api_image_request import api_image_request


class PictureDescriptionApiModel(PictureDescriptionBaseModel):
    # elements_batch_size = 4

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionApiOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionApiOptions

        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True."
                )

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        for image in images:
            yield api_image_request(
                image=image,
                prompt=self.options.prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
                **self.options.params,
            )

    def _annotate_with_context(self, images: Iterable[Image.Image], contexts: List[str]) -> Iterable[str]:
        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        for image, context in zip(images, contexts):
            # Create context-aware prompt
            context_prompt = f"{context}\n\n{self.options.prompt}\nConsider the text context provided above when describing the image."
            
            yield api_image_request(
                image=image,
                prompt=context_prompt,
                url=self.options.url,
                timeout=self.options.timeout,
                headers=self.options.headers,
                **self.options.params,
            )
