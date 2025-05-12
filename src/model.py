"""
Defines the transformer-based classifier for mental health sentiment analysis.
"""

from typing import Tuple
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer




class MentalHealthClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(MentalHealthClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use the CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return self.classifier(cls_output)



def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load tokenizer for the transformer model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        AutoTokenizer: HuggingFace tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)
