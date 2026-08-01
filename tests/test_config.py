from stochastic_retrieval.config import load_config, load_sweep


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


def test_config_parses_independent_sample_counts(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: {name: example/model}",
                "dataset: {name: example, ir_dataset_id: example/test}",
                "experiment:",
                "  name: multi-n",
                "  query_samples: 128",
                "  sample_counts: [1, 2, 4, 8, 16, 32, 64, 128]",
                "  independent_sample_banks: true",
            ]
        )
    )

    config = load_config(config_path)

    assert config.experiment.resolved_sample_counts == (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    )
    assert config.experiment.independent_sample_banks is True


def test_sweep_expands_model_dataset_cartesian_product(tmp_path) -> None:
    (tmp_path / "model.yaml").write_text("name: example/model\n")
    (tmp_path / "first.yaml").write_text(
        "name: first\nir_dataset_id: beir/first/test\n"
    )
    (tmp_path / "second.yaml").write_text(
        "name: second\nir_dataset_id: beir/second/test\n"
    )
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join(
            [
                "name: test-sweep",
                "model_configs: [model.yaml]",
                "dataset_configs: [first.yaml, second.yaml]",
                "experiment:",
                "  query_samples: 4",
            ]
        )
    )

    runs = load_sweep(sweep_path)

    assert len(runs) == 2
    assert {run.dataset.name for run in runs} == {"first", "second"}
    assert all(run.experiment.query_samples == 4 for run in runs)


def test_sweep_expands_model_overrides(tmp_path) -> None:
    (tmp_path / "model.yaml").write_text(
        "name: example/model\ndropout_scope: all\ndropout_probability: null\n"
    )
    (tmp_path / "dataset.yaml").write_text(
        "name: example\nir_dataset_id: beir/example/test\n"
    )
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        "\n".join(
            [
                "name: dropout",
                "model_configs: [model.yaml]",
                "dataset_configs: [dataset.yaml]",
                "model_overrides:",
                "  - label: attention",
                "    values: {dropout_scope: attention}",
                "  - label: p020",
                "    values: {dropout_probability: 0.20}",
                "experiment: {}",
            ]
        )
    )

    runs = load_sweep(sweep_path)

    assert len(runs) == 2
    assert runs[0].model.dropout_scope == "attention"
    assert runs[1].model.dropout_probability == 0.20
    assert runs[1].tags["model_override"] == "p020"
