"""Candidate generation strategies using in-memory recommendation data."""

from similarity import SimilarityCalculator


class CandidateGenerator:
    """Generate bounded recommendation candidate lists from plain dictionaries."""

    def __init__(
        self,
        user_history=None,
        user_profiles=None,
        item_tags=None,
        item_popularity=None,
        limit=20,
    ):
        self.user_history = user_history or {}
        self.user_profiles = user_profiles or {}
        self.item_tags = item_tags or {}
        self.item_popularity = item_popularity or {}
        self.limit = max(1, limit)
        self.similarity = SimilarityCalculator()

    def collaborative_candidates(self, user_id):
        """Recommend items liked by users with similar profile vectors."""
        profile = self.user_profiles.get(user_id)
        if profile is None:
            return self.popularity_candidates()
        scored = []
        for other_id, other_profile in self.user_profiles.items():
            if other_id == user_id:
                continue
            similarity = self.similarity.cosine_similarity(profile, other_profile)
            for item_id in self.user_history.get(other_id, []):
                if item_id not in self.user_history.get(user_id, []):
                    scored.append((similarity, item_id))
        return self._rank_unique(scored)

    def content_based_candidates(self, user_id):
        """Recommend unseen items whose tags overlap with the user's history."""
        history = set(self.user_history.get(user_id, []))
        if not history:
            return self.popularity_candidates()
        liked_tags = set().union(*(set(self.item_tags.get(item, [])) for item in history))
        scored = []
        for item_id, tags in self.item_tags.items():
            if item_id in history:
                continue
            score = self.similarity.jaccard_similarity(liked_tags, tags)
            if score:
                scored.append((score, item_id))
        return self._rank_unique(scored)

    def popularity_candidates(self):
        """Return the most popular items, highest score first."""
        ordered = sorted(self.item_popularity.items(), key=lambda pair: pair[1], reverse=True)
        return [item_id for item_id, _ in ordered[: self.limit]]

    def hybrid_candidates(self, user_id):
        """Merge collaborative, content, and popularity candidates without duplicates."""
        result = []
        for candidates in (
            self.collaborative_candidates(user_id),
            self.content_based_candidates(user_id),
            self.popularity_candidates(),
        ):
            for item_id in candidates:
                if item_id not in result:
                    result.append(item_id)
                if len(result) >= self.limit:
                    return result
        return result

    def _rank_unique(self, scored_items):
        """Sort scored pairs and retain the best score for each item."""
        best_scores = {}
        for score, item_id in scored_items:
            best_scores[item_id] = max(score, best_scores.get(item_id, 0.0))
        ranked = sorted(best_scores.items(), key=lambda pair: pair[1], reverse=True)
        return [item_id for item_id, _ in ranked[: self.limit]]


if __name__ == "__main__":
    generator = CandidateGenerator(
        user_history={"u1": ["book-a"], "u2": ["book-b"]},
        user_profiles={"u1": [1, 0], "u2": [0.9, 0.1]},
        item_tags={"book-a": {"python"}, "book-b": {"python", "data"}, "book-c": {"history"}},
        item_popularity={"book-b": 10, "book-c": 5},
    )
    print("Hybrid candidates:", generator.hybrid_candidates("u1"))
