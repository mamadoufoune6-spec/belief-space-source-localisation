# Release validation — 3 November 2026

The existing project baseline ran 106 tests successfully with its optional
local RadioMapSeer subset present (2.553 seconds). The public release was then
checked without copying that data.

## Installed release check

A local venv with access to the existing installed dependencies was created.
The release itself was installed editable using:

```powershell
python -m pip install --no-deps --no-build-isolation -e .
python -B -m unittest discover -s tests -v
python -B examples/plot_example.py --seed 17
python -B examples/run_greedy.py
python -B examples/information_example.py
```

The test suite discovered 106 tests: **101 passed and 5 were explicitly
skipped**, with no failures/errors (1.528 seconds). The five skips require the
optional RadioMapSeer data not distributed in this release. They are not
counted as passes. Other radio-controller tests use small in-memory fixtures.

The editable installation succeeded using existing dependency packages. This
does not claim a fresh network dependency installation or cross-platform check.

## Actual examples

- The seed-17 prescribed-path example saved `outputs/example.png`. It reported
  prior entropy 6.438 nats, final posterior entropy 1.683 nats and reaching
  success. The plotted path was fixed before observing measurements; this is
  not search-controller benchmark evidence. The README figure is copied from
  this generated output without editing.
- The greedy example used sensor seed 17 and controller seed 3. It declared
  `(10, 8)` with posterior probability 0.592170580067132 after 42 movement
  attempts, budget used 42 and no failed confirmations. Final reaching was true.
  This is one deterministic demonstration, not a success-rate estimate.
- The information example runs one offline action selection and sampling
  diagnostics, using no physical environment advance. Its runtime observations
  are not used as a hardware-performance or controller-superiority claim.

All released Python files match their source-manifest hashes. No controller,
sensor, stopping/budget logic or scientific parameter was changed for release.
The only dependency metadata change makes the already-used Pillow import
explicit. Broader historical benchmarks and experimental research interpretation
were not rerun or added in this release.
