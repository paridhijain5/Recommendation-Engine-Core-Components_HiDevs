"""Minimal standard-library test runner for the recommendation components."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from candidate_gen import CandidateGenerator
from evaluator import RecommendationEvaluator
from scorer import RecommendationScorer
from similarity import SimilarityCalculator


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    try:
        assert condition
        print(f"PASS: {name}")
        passed += 1
    except (AssertionError, Exception) as error:
        print(f"FAIL: {name} ({error})")
        failed += 1


# Similarity checks include ordinary and degenerate inputs.
check("cosine similarity", math.isclose(SimilarityCalculator.cosine_similarity([1, 0], [1, 0]), 1))
check("zero-vector cosine", SimilarityCalculator.cosine_similarity([0, 0], [1, 2]) == 0)
check("empty Jaccard", SimilarityCalculator.jaccard_similarity(set(), set()) == 1)
check("Pearson correlation", math.isclose(SimilarityCalculator.pearson_correlation([1, 2], [2, 4]), 1))

# Candidate generation checks cover similarity, cold start, and deduplication.
generator = CandidateGenerator(
    user_history={"u1": ["a"], "u2": ["b"]},
    user_profiles={"u1": [1, 0], "u2": [0.9, 0.1]},
    item_tags={"a": {"python"}, "b": {"python"}, "c": {"history"}},
    item_popularity={"b": 10, "c": 5},
    limit=2,
)
check("collaborative candidates", generator.collaborative_candidates("u1") == ["b"])
check("cold-start fallback", generator.content_based_candidates("new") == ["b", "c"])
check("hybrid candidates are bounded and unique", len(generator.hybrid_candidates("u1")) <= 2 and len(set(generator.hybrid_candidates("u1"))) == len(generator.hybrid_candidates("u1")))

# Scorer checks cover weighted scores, ranking, and no registered functions.
scorer = RecommendationScorer()
check("zero scorers", scorer.calculate_score("u", "i", {})[0] == 0)
scorer.add_scorer("relevance", lambda user, item, ctx: ctx.get(item, 0), 1)
check("weighted score", scorer.calculate_score("u", "i", {"i": 0.75})[0] == 0.75)
check("candidate ranking", scorer.rank_candidates("u", ["low", "high"], 2, {"low": 0.2, "high": 0.9})[0][0] == "high")

# Evaluation checks cover perfect metrics and missing ground truth.
evaluator = RecommendationEvaluator()
check("precision at k", evaluator.precision_at_k(["a", "b"], {"a"}, 1) == 1)
check("recall at k", evaluator.recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5)
check("empty evaluation", evaluator.evaluate_all({"u1": ["a"]}, {}, 1)["users_evaluated"] == 0)

print(f"\nSummary: {passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
