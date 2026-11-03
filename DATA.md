# Optional RadioMapSeer input

The self-contained analytic examples use no downloaded dataset. Radio-map
modules can read the separately acquired **original RadioMapSeer** release.
No radio maps or copied author loader code are distributed here.

- [Official dataset page](https://radiomapseer.github.io/)
- [Dataset of Pathloss and ToA Radio Maps With Localization Application](https://arxiv.org/abs/2212.11777), Ç. Yapar, R. Levie, G. Kutyniok and G. Caire, 2022.

Obtain data from its authors, follow its terms and cite the dataset paper.
The MIT licence of this repository does not grant dataset rights. Later
RadioMap3DSeer/RadioLocSeer releases are not interchangeable with this input.

Existing integration tests expect an optional audited subset at
`data/radiomapseer-subset/`, with a `manifest.json` and original relative paths:

```text
gain/IRT2/<map>_<transmitter>.png
gain/DPM/<map>_<transmitter>.png
png/buildings_complete/<map>.png
png/antennas/<map>_<transmitter>.png
antenna/<map>.json
```

The historical subset uses maps 1–3 with 80 transmitter candidates per map.
Its manifest and data are not bundled. Tests that require them skip explicitly;
fixture-based radio-controller tests remain runnable without the dataset.

These maps are simulated propagation fields, not physical RSSI measurements.
The code decodes original grayscale path gain to a declared RSSI surrogate,
treating zero pixels as a clipped floor. Coordinates follow the original JSON
frame: array column = x, row = 255 − y. Buildings block motion; candidate
reachability and uncertainty are separate questions. Do not silently change
the source candidates, observation encoding or noise assumptions to improve
apparent results.
