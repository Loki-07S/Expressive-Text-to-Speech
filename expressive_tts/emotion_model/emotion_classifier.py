"""
Emotion Detection Module

Uses HuggingFace bert-base-uncased to detect emotion and intensity from text.
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

class EmotionDetector:
    def __init__(self, model_name="bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Using a model fine-tuned on GoEmotions or similar dataset is recommended
        # For demonstration, we load the base model but note it needs fine-tuning
        # (or loading a pre-finetuned checkpoint like 'bhadresh-savani/bert-base-uncased-emotion')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # We assume 6 basic emotions + neutral for this example
        # Map: 0:sad, 1:happy, 2:anger, 3:surprise, 4:fear, 5:disgust, 6:neutral
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {
            0: "sad",
            1: "happy",
            2: "anger",
            3: "surprise",
            4: "fear",
            5: "disgust",
            6: "neutral"
        }

    def detect_emotion(self, text):
        """
        Detects emotion and intensity from input text.
        Intensity is the max softmax probability.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Get max probability and corresponding label
            intensity, pred_class = torch.max(probs, dim=-1)
            
        emotion = self.label_map[pred_class.item()]
        
        return {
            "emotion": emotion,
            "intensity": intensity.item()
        }

if __name__ == "__main__":
    # Test
    detector = EmotionDetector(model_name="bhadresh-savani/bert-base-uncased-emotion") # A real pre-trained one
    # Note: If you use a different model, update the label_map accordingly. The above is a placeholder structure.
    print(detector.detect_emotion("I am so incredibly sad about this."))
