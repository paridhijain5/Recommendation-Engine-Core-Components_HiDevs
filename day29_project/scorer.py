"""Weighted scoring and ranking for recommendation candidates."""


class RecommendationScorer:
    """Combine named scoring functions into a normalized recommendation score."""

    def __init__(self):
        self.scorers = {}

    def add_scorer(self, name, function, weight):
        """Register or replace a scorer with a non-negative weight."""
        if weight < 0:
            raise ValueError("scorer weights cannot be negative")
        self.scorers[name] = (function, weight)

    def calculate_score(self, user_id, item_id, context):
        """Return ``(score, explanation)`` for one user-item pair."""
        if not self.scorers:
            return 0.0, "No scoring factors registered"
        total_weight = sum(weight for _, weight in self.scorers.values())
        if total_weight == 0:
            return 0.0, "No scoring weights configured"
        contributions = []
        for name, (function, weight) in self.scorers.items():
            value = max(0.0, min(1.0, float(function(user_id, item_id, context))))
            contributions.append((value * weight / total_weight, name))
        score = sum(value for value, _ in contributions)
        top_factor = max(contributions, key=lambda pair: pair[0])[1]
        return max(0.0, min(1.0, score)), f"Top factor: {top_factor}"

    def rank_candidates(self, user_id, candidates, limit, context=None):
        """Return ``(item_id, score, explanation)`` tuples in descending order."""
        ranked = []
        context = context or {}
        for item_id in candidates:
            score, explanation = self.calculate_score(user_id, item_id, context)
            ranked.append((item_id, score, explanation))
        ranked.sort(key=lambda result: result[1], reverse=True)
        return ranked[: max(0, limit)]


if __name__ == "__main__":
    scorer = RecommendationScorer()
    scorer.add_scorer("relevance", lambda user, item, ctx: ctx.get("relevance", {}).get(item, 0), 2)
    scorer.add_scorer("recency", lambda user, item, ctx: ctx.get("recency", {}).get(item, 0), 1)
    scorer.add_scorer("popularity", lambda user, item, ctx: ctx.get("popularity", {}).get(item, 0), 1)
    context = {"relevance": {"a": 1}, "recency": {"a": 0.5}, "popularity": {"a": 0.8}}
    print(scorer.calculate_score("user-1", "a", context))
