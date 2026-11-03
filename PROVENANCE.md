# Provenance

This release is extracted from Mamadou's existing belief-space-planning project
on 3 November 2026. The original project remains intact. All released Python
files were copied byte-for-byte; the manifest records their source hashes.

Mamadou selected the localisation problem, modelling choices, controller
objectives, comparisons and research direction. Existing project records
document substantial Codex implementation and testing from researcher-guided
designs, including radio-map adapters, routing, dual-control variants and
benchmark infrastructure. This release does not claim that all implementation
was independently written by Mamadou.

Codex prepared the release README, model/data documentation, source manifest,
licence and validation record. Pillow was made an explicit dependency because
existing radio-map code/tests import it. No scientific parameters, algorithms,
stopping logic, seeds or controller objectives were changed.

Only the self-contained example was rerun for its public figure. Historical
benchmark output and downloaded data are not republished, and their conclusions
are not reasserted as new results. Research interpretation and further
experiments remain the researcher's responsibility.

## CI addition, 3 November 2026

At Mamadou's request, Codex authored the GitHub Actions workflow and associated
README/validation notes. It installs the package, runs existing tests without
downloading optional RadioMapSeer data, and reports actual passes, failures,
errors and skips separately in logs and the job summary. Local verification in
the existing installed project environment reported 101 passes and five
optional-data skips. The existing source, tests, skip conditions and scientific
behaviour are unchanged. The workflow does not run research benchmarks or turn
skipped real-data tests into passing checks.
