"""Small, dependency-free similarity measures for recommendation systems."""

import math


class SimilarityCalculator:
    """Calculate common similarity and correlation metrics."""

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Return cosine similarity, mapped to the range 0..1.

        Negative cosine values are clipped because recommendation similarity is
        generally interpreted as a non-negative strength.
        """
        if len(vec1) != len(vec2):
            raise ValueError("vectors must have the same length")
        if not vec1:
            return 0.0
        dot = sum(left * right for left, right in zip(vec1, vec2))
        norm1 = math.sqrt(sum(value * value for value in vec1))
        norm2 = math.sqrt(sum(value * value for value in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cosine = dot / (norm1 * norm2)
        return max(0.0, min(1.0, cosine))

    @staticmethod
    def jaccard_similarity(set1, set2):
        """Return intersection-over-union for two sets."""
        first, second = set(set1), set(set2)
        union = first | second
        return len(first & second) / len(union) if union else 1.0

    @staticmethod
    def pearson_correlation(ratings1, ratings2):
        """Return Pearson correlation, or zero when variance is undefined."""
        if len(ratings1) != len(ratings2):
            raise ValueError("rating sequences must have the same length")
        if not ratings1:
            return 0.0
        mean1 = sum(ratings1) / len(ratings1)
        mean2 = sum(ratings2) / len(ratings2)
        centered1 = [value - mean1 for value in ratings1]
        centered2 = [value - mean2 for value in ratings2]
        numerator = sum(left * right for left, right in zip(centered1, centered2))
        denominator = math.sqrt(sum(value * value for value in centered1)) * math.sqrt(
            sum(value * value for value in centered2)
        )
        return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    calculator = SimilarityCalculator()
    print("Cosine:", calculator.cosine_similarity([1, 0], [1, 1]))
    print("Jaccard:", calculator.jaccard_similarity({"python", "ml"}, {"ml", "api"}))
    print("Pearson:", calculator.pearson_correlation([1, 2, 3], [2, 4, 6]))
