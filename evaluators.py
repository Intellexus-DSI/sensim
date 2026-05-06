from sentence_transformers.evaluation import SentenceEvaluator
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm


class CrossEncoderCorrelationEvaluator(SentenceEvaluator):
    """
    Replacement for old CrossEncoderCorrelationEvaluator.
    Computes Pearson & Spearman correlation between predicted scores and gold labels.
    """

    def __init__(self, sentences1, sentences2, scores, name=""):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = np.array(scores, dtype=float)
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        pred_scores = []

        for s1, s2 in tqdm(zip(self.sentences1, self.sentences2), total=len(self.sentences1), desc="Evaluating"):
            pred = model.predict([s1, s2])
            pred_scores.append(pred)

        pred_scores = np.array(pred_scores, dtype=float)

        pear, _ = pearsonr(pred_scores, self.scores)
        spear, _ = spearmanr(pred_scores, self.scores)
        kend, _ = kendalltau(pred_scores, self.scores)

        print(f"Eval {self.name} — Pearson: {pear:.4f}, Spearman: {spear:.4f}, Kendall: {kend:.4f}")
        return pear  # higher is better
