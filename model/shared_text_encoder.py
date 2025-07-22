import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from typing import List, Optional

class TextEncoder(nn.Module):
    def __init__(self, 
                 pretrained_model_name: str = "roberta-base",
                 mlp_hidden_size: int = 640,
                 output_dim: int = 512,
                 max_length: int = 77):
        super().__init__()

        #1) 
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.max_length=max_length

        hidden_size = self.roberta.config.hidden_size
        self.proj1= nn.Linear(hidden_size, mlp_hidden_size)
        self.reru = nn.ReLU()
        self.proj2=nn.Linear(mlp_hidden_size, output_dim)

    def forward(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).to(self.roberta.device)
        pooled = self.roberta(**inputs).pooler_output
        #print(f"pooled shape: {pooled.shape}")
        x = self.proj1(pooled)
        x = self.reru(x)
        x = self.proj2(x)
        #print(f"output shape: {x.shape}")
        return x
    


if __name__ == "__main__":
    # Example usage
    encoder = TextEncoder()
    texts = ["Hello, world!","da"]
    embeddings = encoder(texts)
    print(embeddings.shape)  # Should print (batch_size, output_dim)
    print(embeddings)  # Should print (batch_size, output_dim)