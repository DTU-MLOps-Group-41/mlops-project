# Testing

This project uses pytest for testing with coverage reporting and GitHub Actions for CI/CD.

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
invoke test

# Or directly with pytest
uv run pytest tests/

# With verbose output
uv run pytest tests/ -v
```

### Coverage Report

```bash
# Run tests and generate coverage report
uv run coverage run -m pytest tests/
uv run coverage report -m

# Generate HTML report
uv run coverage html
# Open htmlcov/index.html in browser
```

## Test Structure

Tests are located in the `tests/` directory:

| File | Tests |
|------|-------|
| `test_data.py` | Dataset loading, preprocessing, tokenization, splits |
| `test_model.py` | Model initialization, forward pass, Lightning integration |
| `test_train.py` | Training pipeline, callbacks, configuration |
| `test_api.py` | API endpoints, request validation, responses |
| `test_evaluate.py` | Model evaluation pipeline |
| `test_visualize.py` | Visualization generation |

## Test Categories

### Data Tests (`test_data.py`)

- Dataset download and caching
- Preprocessing pipeline
- Tokenization correctness
- Train/val/test split ratios
- Label encoding
- Parquet file I/O

### Model Tests (`test_model.py`)

- Model instantiation
- Forward pass shapes
- Loss computation
- Metric computation
- Checkpoint save/load

### Training Tests (`test_train.py`)

- Trainer initialization
- Callback behavior
- Early stopping
- Checkpointing
- Hydra configuration loading

### API Tests (`test_api.py`)

- `GET /` endpoint
- `GET /health` endpoint
- `POST /predict` with valid input
- Input validation (empty text, missing fields)
- Response schema validation

## CI/CD Pipeline

Tests run automatically on GitHub Actions for every push and pull request to `main`.

### Workflow Configuration

The workflow is defined in `.github/workflows/tests.yaml`:

- **Triggers**: Push to main, PRs to main, manual dispatch
- **Platforms**: Ubuntu, Windows, macOS
- **Python**: 3.12
- **Coverage**: Uploaded to Codecov

### Pipeline Steps

1. Checkout code
2. Install UV and dependencies
3. Run pytest with coverage
4. Upload coverage report to Codecov

### Concurrency

The workflow uses concurrency groups to cancel outdated runs:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## Writing Tests

### Conventions

- Test files: `test_<module>.py`
- Test functions: `test_<functionality>()`
- Use pytest fixtures for shared setup
- Mock external services (Kaggle, W&B) where possible

### Example Test

```python
import pytest
from customer_support.model import TicketClassificationModule

def test_model_forward_pass():
    model = TicketClassificationModule(num_classes=3)

    # Create dummy input
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)

    # Forward pass
    outputs = model(input_ids, attention_mask)

    # Check output shape
    assert outputs.logits.shape == (2, 3)
```

### Fixtures

Common fixtures are defined in `tests/__init__.py` or individual test files:

```python
@pytest.fixture
def sample_dataset():
    """Create a small test dataset."""
    return TicketDataset(root="data", split="train", dataset_type="small")
```

## Code Quality

In addition to tests, the project uses:

- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Pre-commit hooks**: Run checks before each commit

See [Getting Started](getting_started.md) for setup instructions.
