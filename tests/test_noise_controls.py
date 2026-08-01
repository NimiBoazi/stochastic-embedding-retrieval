import numpy as np

from stochastic_retrieval.noise_controls import (
    evaluate_matched_noise_oracles,
    matched_noise_controls,
)
from stochastic_retrieval.retrieval import DenseRetriever, l2_normalize


def test_matched_controls_preserve_every_dropout_angle_and_norm() -> None:
    deterministic = l2_normalize(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    )
    stochastic = l2_normalize(
        np.array(
            [
                [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0]],
                [[1.0, 0.0, 0.4], [0.0, 1.0, 0.4]],
                [[1.0, -0.3, 0.1], [-0.2, 1.0, 0.2]],
            ],
            dtype=np.float32,
        )
    )

    controls, diagnostics = matched_noise_controls(
        deterministic,
        stochastic,
        ["q1", "q2"],
        seed=9,
    )
    expected_angles = np.arccos(
        np.clip(np.einsum("sqd,qd->sq", stochastic, deterministic), -1.0, 1.0)
    )
    expected_chords = np.linalg.norm(stochastic - deterministic, axis=2)

    for control in controls.values():
        np.testing.assert_allclose(np.linalg.norm(control, axis=2), 1.0, atol=1e-6)
        actual_angles = np.arccos(
            np.clip(np.einsum("sqd,qd->sq", control, deterministic), -1.0, 1.0)
        )
        actual_chords = np.linalg.norm(control - deterministic, axis=2)
        np.testing.assert_allclose(actual_angles, expected_angles, atol=1e-6)
        np.testing.assert_allclose(actual_chords, expected_chords, atol=1e-6)

    assert len(diagnostics) == 4
    assert diagnostics["covariance_estimable"].all()
    assert diagnostics["maximum_angle_mismatch"].max() < 1e-6


def test_matched_controls_are_reproducible_and_handle_n1() -> None:
    deterministic = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    stochastic = l2_normalize(
        np.array([[[1.0, 0.2, 0.1]]], dtype=np.float32)
    )

    first, diagnostics = matched_noise_controls(
        deterministic,
        stochastic,
        ["q1"],
        seed=4,
    )
    second, _ = matched_noise_controls(
        deterministic,
        stochastic,
        ["q1"],
        seed=4,
    )

    for method in first:
        np.testing.assert_array_equal(first[method], second[method])
    assert not diagnostics["covariance_estimable"].any()


def test_noise_oracle_evaluation_keeps_candidate_provenance_separate() -> None:
    documents = l2_normalize(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    )
    deterministic = l2_normalize(
        np.array([[1.0, 0.2, 0.0]], dtype=np.float32)
    )
    stochastic = l2_normalize(
        np.array(
            [
                [[1.0, 0.3, 0.1]],
                [[1.0, 0.1, 0.3]],
            ],
            dtype=np.float32,
        )
    )
    retriever = DenseRetriever(documents, k=3, backend="numpy")
    deterministic_ranking = retriever.search(deterministic)
    dropout_rankings = [retriever.search(sample) for sample in stochastic]

    per_query, selections, diagnostics = evaluate_matched_noise_oracles(
        deterministic,
        stochastic,
        deterministic_ranking,
        dropout_rankings,
        retriever,
        ["q1"],
        ["d1", "d2", "d3"],
        {"q1": {"d1": 1}},
        metric_cutoffs=(1,),
        success_cutoffs=(1,),
        sample_count=2,
        seed=12,
        selection_cutoff=1,
    )

    assert set(per_query["method"]) == {
        "deterministic",
        "dropout_oracle",
        "full_covariance_gaussian_oracle",
        "isotropic_noise_oracle",
    }
    assert set(selections["candidate_source"]) == {
        "dropout_stochastic",
        "full_covariance_gaussian_oracle",
        "isotropic_noise_oracle",
    }
    assert len(diagnostics) == 2
