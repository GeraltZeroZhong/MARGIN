# Synthetic test fixtures

Run the tests from the repository root with `python -m pytest`. The suite uses
small deterministic inputs and temporary output directories. Model execution,
external datasets, network access, and GPU hardware are unnecessary.

`configs/synthetic.yaml` supplies demo parameters. `margin.fixtures` generates
sequences from a seeded random generator and coordinates from trigonometric
functions. Residue annotations follow explicit repeating patterns. Toy benchmark
relations are constructed from those generated records. Toy mutation effects are
derived from the synthetic teacher scores and deterministic sinusoidal offsets;
they test software behavior and have no independent experimental interpretation.

The remaining fixtures are constructed directly in each test:

- Parser inputs use invented identifiers, short alphabet fragments, and tabular
  values chosen to test normalization and duplicate handling.
- Metric tests use small numeric arrays with analytical expected results and a
  deliberately unequal group-size example.
- Structure preprocessing tests use a tiny annotation mapping and a sequence
  mismatch to check residue alignment behavior.
- Registry, state-bank, and decoy tests check leakage labels, reproducibility,
  state coverage, and contact-graph degree preservation.
- Teacher tests check normalized probabilities, complete position coverage,
  mutation schema errors, and reusable cache behavior. Cache fixtures contain
  placeholder bytes and zero scores; subprocess execution is intercepted.
- Decision tests construct pass/fail criterion tables to exercise each branch.
- The pipeline test runs the complete synthetic workflow and checks output
  schemas, exclusion roles, report links, and the configured confidence label.

The example run is available through
`python -m margin.cli run --config configs/synthetic.yaml`.
