import gc
import torch

from enum import Enum
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)

from app.captioner import CaptioningModels

# Label mapping
id2label = {
  0: 'Cultural_Religious',
  1: 'Fauna_Flora',
  2: 'Gastronomy',
  3: 'Nature',
  4: 'Sports',
  5: 'Urban_Rural'
}

class ClassifierModels(str, Enum):
    bert = 'bert'
    distilbert = 'distilbert'
    roberta = 'roberta'

class Classifier:
    def __init__(self):
        self.classifier_model_type = None
        self.captioner_model_type = None
        self.model = None
        self.tokenizer= None
        self.device = 'cuda'

    def clear_model(self) -> None:
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect
        self.model = None
        self.classifier_model_type = None
        self.captioning_model_type = None
    
    def _initialize_model(self, classifier_model_type: str, captioning_model_type: str):
        classifier_model_type = classifier_model_type.lower()
        captioning_model_type = captioning_model_type.lower()
        assert classifier_model_type in ClassifierModels.__members__
        assert captioning_model_type in CaptioningModels.__members__
        if self.model is not None:
            if (self.classifier_model_type != classifier_model_type 
                or self.captioning_model_type != captioning_model_type):
                self.clear_model()
            else:
                return
        model_path = f"models/{classifier_model_type}_{captioning_model_type}"
        self.classifier_model_type = classifier_model_type
        self.captioning_model_type = captioning_model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set model to evaluation mode

    def predict(
        self, 
        caption: str,
        classifier_model_type: str = ClassifierModels.distilbert,
        captioning_model_type: str = CaptioningModels.blip
    ):
        self._initialize_model(classifier_model_type, captioning_model_type)

        # Tokenize the caption
        inputs = self.tokenizer(caption, return_tensors="pt", truncation=True)

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label_id = torch.argmax(predictions, dim=1).item()
            predicted_label = id2label[predicted_label_id]

        return predicted_label





