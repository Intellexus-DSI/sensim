from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class BWSnTupleLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=util.pairwise_cos_sim) -> None:

        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scale = scale

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # 1. Extract embeddings: List of [Batch, Dim]
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        # 2. Calculate pairwise similarity from embeddings[0] to all others
        # We loop through embeddings[1...n] and calculate similarity against embeddings[0]
        # Each 'sim' will be shape [Batch]
        anchor = embeddings[0]
        all_similarities = [self.similarity_fct(anchor, variant) for variant in embeddings[1:]]

        # This represents the total "strength" of the anchor against the n-1 variants
        scores = torch.stack(all_similarities)  # Shape: [n-1, Batch]

        scores = scores * self.scale

        # 4. Batch-wise comparison (Broadcasting)
        # This compares the 'total strength' of sample i vs sample j
        scores_diff = scores[:, None] - scores[None, :]

        # 5. Label matrix: which sample in the batch is "better" ground truth
        label_mask = (labels[:, None] < labels[None, :]).float()

        # 6. Masking: suppress pairs where label j is not > label i
        scores_diff = scores_diff - (1 - label_mask) * 1e12

        # 7. LogSumExp for loss calculation
        # Flatten and append 0 (e^0 = 1) for the denominator of the softmax-like structure
        flattened_scores = scores_diff.view(-1)
        combined_scores = torch.cat((torch.zeros(1, device=scores.device), flattened_scores), dim=0)

        loss = torch.logsumexp(combined_scores, dim=0)

        return loss

    def forward_2(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        scores = self.similarity_fct(embeddings[0], embeddings[1])
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return 'TBD'

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
    loss = BWSnTupleLoss(model)

    # Use raw strings and tokenize them correctly
    sentences = ['hello world', 'hi', 'good morning', 'nice to meet', 'bye']
    sentence_features = [model.tokenize([s]) for s in sentences]

    # Ensure tensors are on the correct device
    sentence_features = [{k: v.to(model.device) for k, v in feature.items()} for feature in sentence_features]

    # labels for the 4 variants compared to the anchor
    labels = torch.tensor([0.9, 0.8, 0.3, 0.1], device=model.device)

    r = loss.forward(sentence_features, labels)

    print(r)

def main_2():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
    loss = BWSnTupleLoss(model)

    # Use raw strings and tokenize them correctly
    sentences = [['hello world', 'hi'],['hello world', 'good morning'],['hello world', 'nice to meet'],['hello world', 'bye']]
    sentence_features = [model.tokenize([s]) for s in sentences]

    # Ensure tensors are on the correct device
    sentence_features = [{k: v.to(model.device) for k, v in feature.items()} for feature in sentence_features]

    # labels for the 4 variants compared to the anchor
    labels = torch.tensor([0.9, 0.8, 0.3, 0.1], device=model.device)
    labels = labels.T

    r = loss.forward_2(sentence_features, labels)

    print(r)

if __name__ == '__main__':
    main()
