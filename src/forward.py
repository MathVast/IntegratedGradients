from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer       

if __name__ == '__main__':
    # Get query-passage
    query = "what was the immediate impact of the success of the manhattan project?"  # "what do the xylem and the phloem do"
    passage = "The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science."
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L12-v2").to(device)
    model.eval()

    num_labels = model.config.num_labels

    inputs = tokenizer(
        query,
        passage,
        max_length=512,
        truncation=True,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model(**inputs)
    
    print(f"** Default model output: {outputs} **")
    print(f"** From logits to proba: {torch.sigmoid(outputs.logits)} **")
    print(f"** From proba to label: {1 if torch.sigmoid(outputs.logits) >= 0.5 else 0} **")
