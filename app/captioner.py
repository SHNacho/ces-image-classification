import gc
import torch

from enum import Enum
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    BlipProcessor, 
    BlipForConditionalGeneration
)

class CaptioningModels(str, Enum):
    llava = 'llava'
    blip = 'blip'

class Captioner:
    def __init__(self):
        self.model_type = None
        self.model = None
        self.processor = None
        self.device = 'cuda'

    def clear_model(self) -> None:
        del self.model
        torch.cuda.empty_cache()
        gc.collect
        self.model = None
        self.model_type = None
        self.processor = None

    def _initialice_model(self, model_type: str):
        model_type = model_type.lower()
        assert model_type in CaptioningModels.__members__
        if self.model is not None:
            if self.model_type != model_type:
                self.clear_model()
            else:
                return
        if model_type == 'llava':
            self._initialice_llava_model()
        elif model_type == 'blip':
            self._initialice_blip_model()

    def generate_caption(self, model_type: str, image: Image):
        self._initialice_model(model_type)
        if model_type == 'llava':
            return self._generate_llava_caption(image=image)
        elif model_type == 'blip':
            return self._generate_blip_caption(image=image)

    def _initialice_llava_model(self):
        llava_model_id = "llava-hf/llava-1.5-7b-hf"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_id,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        self.processor = AutoProcessor.from_pretrained(llava_model_id)
        self.model_type = 'llava-1.5'

    def _initialice_blip_model(self):
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", 
        ).to(self.device)
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large")
        self.model_type = 'llava-1.5'

    def _generate_llava_caption(
        self,
        image: Image,
        prompt: str = "USER: <image>\nWhat is shown in this image? ASSISTANT:",
    ):
        inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
                ).to(self.device, torch.float16)
        out = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
        caption = self.processor.decode(out[0][2:], skip_special_tokens=True)
        print(f"Generated caption:", caption)
        return caption.replace('ER:  \nWhat is shown in this image? ASSISTANT: ', '')

    def _generate_blip_caption(
        self,
        image: Image,
    ):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"Generated caption: {caption}")
        return caption
