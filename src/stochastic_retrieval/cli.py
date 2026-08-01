from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from stochastic_retrieval.config import load_config, load_sweep
from stochastic_retrieval.pipeline import run_experiment

app = typer.Typer(
    no_args_is_help=True,
    help="Run reproducible stochastic embedding retrieval experiments.",
)


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to a complete YAML experiment configuration.",
        ),
    ],
) -> None:
    """Encode, retrieve, aggregate, evaluate, and persist one experiment."""
    config = load_config(config_path)
    project_root = Path(__file__).resolve().parents[2]
    store = run_experiment(config, project_root)
    typer.echo(f"Completed run: {store.run_id}")
    typer.echo(f"Artifacts: {store.run_dir}")


@app.command()
def validate(
    config_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Parse a configuration without loading models or datasets."""
    config = load_config(config_path)
    typer.echo(json.dumps(asdict(config), indent=2))
    typer.echo(f"fingerprint: {config.fingerprint}")


@app.command()
def sweep(
    sweep_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Run every model-dataset condition; otherwise only list them.",
        ),
    ] = False,
) -> None:
    """List or sequentially execute a model-by-dataset experiment matrix."""
    configs = load_sweep(sweep_path)
    typer.echo(f"Sweep contains {len(configs)} independent runs:")
    for config in configs:
        typer.echo(
            f"- {config.experiment.name}: "
            f"{config.model.name} × {config.dataset.ir_dataset_id}"
        )
    if not execute:
        typer.echo("Dry run only. Pass --execute to start the sweep.")
        return

    project_root = Path(__file__).resolve().parents[2]
    failures: list[tuple[str, str]] = []
    for index, config in enumerate(configs, start=1):
        typer.echo(f"\n[{index}/{len(configs)}] Starting {config.experiment.name}")
        try:
            run_experiment(config, project_root)
        except Exception as exc:  # The failed run writes its own failure manifest.
            failures.append((config.experiment.name, str(exc)))
            typer.echo(f"FAILED: {config.experiment.name}: {exc}", err=True)
    if failures:
        typer.echo(f"\n{len(failures)} sweep runs failed:", err=True)
        for name, error in failures:
            typer.echo(f"- {name}: {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"\nCompleted all {len(configs)} sweep runs.")


if __name__ == "__main__":
    app()
