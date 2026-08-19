import pytest

from app.llm.embeddings import _cosine_similarity


def test_identical_vectors_have_similarity_one():
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_zero_vector_returns_zero_instead_of_dividing_by_zero():
    assert _cosine_similarity([0, 0], [1, 1]) == 0.0


def test_mismatched_dimensions_raise_instead_of_silently_truncating():
    with pytest.raises(ValueError):
        _cosine_similarity([1, 2, 3], [1, 2])
