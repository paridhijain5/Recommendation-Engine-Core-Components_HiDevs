"""Offline evaluation metrics for ranked recommendation lists."""

import math


class RecommendationEvaluator:
    """Calculate precision, recall, and rank-aware nDCG metrics."""

    @staticmethod
    def precision_at_k(recommendations, relevant_items, k):
        """Return the fraction of the first ``k`` recommendations that are relevant."""
        top = list(recommendations)[: max(0, k)]
        if not top:
            return 0.0
        relevant = set(relevant_items)
        return sum(item in relevant for item in top) / len(top)

    @staticmethod
    def recall_at_k(recommendations, relevant_items, k):
        """Return the fraction of relevant items found in the first ``k`` results."""
        relevant = set(relevant_items)
        if not relevant:
            return 0.0
        top = set(list(recommendations)[: max(0, k)])
        return len(top & relevant) / len(relevant)

    @staticmethod
    def ndcg_at_k(recommendations, relevant_items, k):
        """Return binary-relevance nDCG with log2 rank discounting."""
        relevant = set(relevant_items)
        if not relevant or k <= 0:
            return 0.0
        top = list(recommendations)[:k]
        dcg = sum(
            1 / math.log2(rank + 2)
            for rank, item in enumerate(top)
            if item in relevant
        )
        ideal_hits = min(k, len(relevant))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
        return dcg / ideal if ideal else 0.0

    def evaluate_all(self, recommendations_dict, ground_truth_dict, k):
        """Average metrics over users present in both input dictionaries."""
        users = [user for user in recommendations_dict if user in ground_truth_dict]
        if not users:
            return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "users_evaluated": 0}
        metrics = [
            (
                self.precision_at_k(recommendations_dict[user], ground_truth_dict[user], k),
                self.recall_at_k(recommendations_dict[user], ground_truth_dict[user], k),
                self.ndcg_at_k(recommendations_dict[user], ground_truth_dict[user], k),
            )
            for user in users
        ]
        return {
            "precision": sum(metric[0] for metric in metrics) / len(users),
            "recall": sum(metric[1] for metric in metrics) / len(users),
            "ndcg": sum(metric[2] for metric in metrics) / len(users),
            "users_evaluated": len(users),
        }


if __name__ == "__main__":
    evaluator = RecommendationEvaluator()
    print(evaluator.evaluate_all({"u1": ["a", "b"]}, {"u1": ["b"]}, 2))
