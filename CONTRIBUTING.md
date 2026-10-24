# Contributing

How to set up a development environment, the checks every change must pass, and the conventions the code follows. Read [ARCHITECTURE.md](ARCHITECTURE.md) first for how the code fits together.

## Set up

Python 3.11 or 3.12 and a webcam (for live testing).

```bash
git clone https://github.com/yagnangg333/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -c constraints.txt -e ".[dev]"
```

On Linux, also install `libgl1 libglib2.0-0 libegl1 libgles2`. Check the setup with `pytest` and `drowsiness dashboard`.

## Checks

Every change must pass the same checks as CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml), on Python 3.11 and 3.12):

```bash
ruff check src tests tools          # lint
ruff format --check src tests tools # formatting (run without --check to fix)
pytest                              # all tests, about 30 s
```

- **Unit tests** drive the classes with synthetic timestamps and values, so timing logic is tested exactly and fast.
- **End-to-end tests** run the real MediaPipe model on the images in [`tests/fixtures/`](tests/fixtures) and on modified copies (darkened, dark glasses, glare, a hand over one eye). They are skipped if the model file is missing.
- Tests check that the code behaves as designed. Whether a change improves *detection* is checked on the datasets with the tools in [`tools/`](tools) (see [EVALUATION.md](EVALUATION.md)).

Add or update a test for every behaviour you change, in the test file for that module (`src/drowsiness/x.py` → `tests/test_x.py`).

## Workflow

1. Branch from `main`: `feat/...` for features, `fix/...` for fixes, `docs/...` or `chore/...` for the rest.
2. Commit small, self-contained changes. The commit message says what changed and, in the body, why.
3. Open a pull request against `main`. CI must pass before merging.
4. Changes to detection settings must say how they were checked: which datasets, and the result on data the setting was not chosen on.

## Conventions

- **Style:** ruff with a line length of 100 ([`pyproject.toml`](pyproject.toml)). Type hints on public functions. Dataclasses for data passed between stages, frozen where they are snapshots.
- **Time in seconds, never frame counts.** Every duration comes from frame timestamps.
- **No magic numbers.** Thresholds, windows and weights belong in [`config.py`](src/drowsiness/config.py) and [`configs/default.toml`](configs/default.toml), which must stay identical (a test checks this), with a comment saying how the value was chosen.
- **Safety first.** Never let a change silence an alert when the driver cannot be observed (face or camera lost), and keep the microsleep rule independent of the fusion.
- **Privacy.** Never write images or video to disk. Datasets, logs and recordings stay out of Git (see [`.gitignore`](.gitignore)).
- **Comments explain why,** not what; docstrings describe what a class or function is for.

## Project layout

See the [repository map in the README](README.md#repository-layout).
