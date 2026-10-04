# Contributing

Use a source checkout with the development dependencies installed as described in
[README.md](README.md). Keep changes focused on reusable software behavior and
include a small synthetic example when an interface changes.

## Local checks

The public tests use deterministic toy inputs and temporary output directories.
Once dependencies are installed, run them without network access or model
downloads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true python -m pytest -q
python -m ruff check src tests scripts/models
```

The end-to-end test runs the synthetic audit on a CPU. Other tests cover schema
validation, structure parsing, sequence states, control generation, cache reuse,
metrics, and decision branches. External model execution requires separately
configured local environments and checkpoints.

## Test data and interfaces

Construct fixtures independently from the documented schema. Use invented
identifiers, small analytical examples, and fixed random seeds. State the fixture
origin in the test or generator. Keep meaningful assertions for units, shapes,
joins, invariants, expected outputs, and invalid inputs.

Preserve numerical behavior when renaming or reorganizing code. Update affected
imports, CLI help, configuration examples, and output readers together. Document
changes to public APIs or artifact names so callers can migrate. Write source
comments, error messages, examples, and documentation in English.

## Building a package

With the build dependencies already available locally:

```bash
python -m build --no-isolation
```

Inspect both the wheel and source archive to confirm that they contain the
intended software, documentation, and synthetic examples. Test installation from
the built wheel in a separate environment before publishing a release.
