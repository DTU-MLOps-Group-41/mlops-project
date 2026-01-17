import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "customer_support"
PYTHON_VERSION = "3.12"


# Project commands
@task
def download_data(ctx: Context) -> None:
    """Download raw data from Kaggle."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py download", echo=True, pty=not WINDOWS)


@task
def preprocess_data(
    ctx: Context,
    dataset_type: str = "",
    length_percentile: int = 0,
    length_handling: str = "trim",
) -> None:
    """Preprocess data.

    Args:
        dataset_type: One of 'small', 'medium', 'full'. If empty, processes all.
        length_percentile: Percentile threshold for sequence length (e.g., 90 for P90). 0 = no filtering.
        length_handling: How to handle long sequences: 'trim' or 'drop'.
    """
    if dataset_type:
        cmd = f"uv run src/{PROJECT_NAME}/data.py preprocess -d {dataset_type}"
    else:
        cmd = f"uv run src/{PROJECT_NAME}/data.py preprocess --all"

    if length_percentile > 0:
        cmd += f" -p {length_percentile}"
    cmd += f" --length-handling {length_handling}"

    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, checkpoint: str) -> None:
    """Evaluate model.

    Args:
        checkpoint: Path to model checkpoint file.
    """
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py {checkpoint}", echo=True, pty=not WINDOWS)

@task
def visualize(ctx: Context, checkpoint: str) -> None:
    """Visualize model results.

    Args:
        checkpoint: Path to model checkpoint file.
    """
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py {checkpoint}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def project_tree(ctx: Context, depth: int = 0) -> None:
    """Generate project structure for README.

    Args:
        depth: Limit directory depth (0 = unlimited)
    """
    depth_flag = f"-L {depth}" if depth > 0 else ""
    ctx.run(
        f"tree -a {depth_flag} -I '.git|.venv|__pycache__|.ruff_cache|.pytest_cache|*.pyc|.gitkeep|.DS_Store' --dirsfirst",
        echo=True,
        pty=not WINDOWS,
    )
