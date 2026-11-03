# Analytic model and evaluation conventions

The source of truth for defaults is `src/belief_search/types.py`.

| Quantity | Existing default |
|---|---|
| Arena / grid | 12 m × 12 m; 0.5 m spacing; 25 × 25 candidates |
| Reference RSSI / distance | −40 dBm at 1 m |
| Path-loss exponent / minimum distance | 2 / 0.5 m |
| Added Gaussian noise | Standard deviation 2 dB |
| Movement | Four cardinal actions; rejected boundary commands consume an attempt |
| Reaching radius / movement-attempt budget | 0.5 m / 80 attempts |

The mean signal is reference power minus `10 * path_loss_exponent * log10(d/d0)`,
with distance clamped below by the minimum distance. Bayesian inference starts
from a uniform grid prior and uses the Gaussian likelihood. Posterior updating
normalises the product in log space; a zero likelihood over the entire supported
prior raises an error rather than silently resetting the belief. Likelihood
underflow before the update remains a numerical limitation.

Belief arrays use `[y_index, x_index]`. Ground truth is accessible to the runner
and plots, not supplied in controller observations. Sensor and planning random
generators are separated.

The prescribed-path example ends under the environment's reaching/attempt rules.
The default greedy runner instead uses its existing declaration/confirmation and
budget rules. An unchanged selected MAP after a WAIT can end in declaration;
this need not be correct. A failed confirmation can incur a separate charge at
the budget boundary. An explicit `stop_on_reach=True` enables first-reach
termination in the relevant runners. These modes must be named when comparing
results; they were not harmonised or altered in this release.

The radio-map experiments use different maps, movement resolution and candidate
banks. Their assumptions and optional input are described in [DATA.md](../DATA.md);
the analytic defaults above are not a specification of those experiments.
