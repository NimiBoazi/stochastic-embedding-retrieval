from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from stochastic_retrieval.config import load_config
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


if __name__ == "__main__":
    app()
