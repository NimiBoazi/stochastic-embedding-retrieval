from stochastic_retrieval.config import load_config


def test_config_supports_referenced_model_and_dataset(tmp_path) -> None:
    models = tmp_path / "models"
    datasets = tmp_path / "datasets"
    experiments = tmp_path / "experiments"
    models.mkdir()
    datasets.mkdir()
    experiments.mkdir()
    (models / "model.yaml").write_text(
        "name: example/model\npooling: mean\nexpected_dimension: 768\n"
    )
    (datasets / "dataset.yaml").write_text(
        "name: example\nir_dataset_id: beir/example/test\n"
    )
    experiment_path = experiments / "experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            [
                "model_config: ../models/model.yaml",
                "dataset_config: ../datasets/dataset.yaml",
                "experiment:",
                "  name: example-run",
            ]
        )
    )

    config = load_config(experiment_path)

    assert config.model.name == "example/model"
    assert config.model.pooling == "mean"
    assert config.dataset.ir_dataset_id == "beir/example/test"
    assert config.experiment.name == "example-run"
