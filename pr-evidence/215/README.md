# Independent metric export review for PR215

Author source: c27e584ae7d018391cb4daeb0d3cd063e16c9757.
Fit-scaling candidate: 6652f6e1c2e6bedfc11fe2d24318cabf1a55ab09 on antonsoo/scikit-gstat.
Python3.12.12, NumPy2.5.4, SciPy1.16.3, pandas3.0.6, GSTools1.7.0.

The unit-summary files come from the independent analytic grid in the candidate's
`skgstat/tests/test_metric_time_units.py`. Expressing a four-year grid in seconds
changes the author's fitted normalized parameters from [30,2,1.7,.2] to
[60,4,1.2928,1.2527]. The normalized candidate recovers the original parameters
and transforms parameter covariance back to physical units.

`metric_probe.py` checks the fitted-model/export contract for Euclidean,
cityblock and squared-Euclidean spatial distances across six marginal models.
Only Euclidean is preserved by the current connector.

`recorded_weather.py` reads the pinned NCEP reanalysis netCDF as data with SciPy.
It selects nine grid locations and24dates from2013-2014, with162conditioning
records and54held-out positions. Coordinates use an explicitly approximate local
equirectangular projection, in km or100km units. This is a numerical contract
check, not a forecast-accuracy or projection-quality study.

After installing the checkout and GSTools1.7.0 in a Python3.12 environment, run:

```sh
python metric_probe.py --source /path/to/c27e584-checkout --output metrics.json
python recorded_weather.py --source /path/to/c27e584-checkout --data /path/to/air_temperature.nc --space-unit-km 1 --output weather.json
```

Repeat the weather command with `--space-unit-km 100` and with the candidate
checkout. The original data's pinned URL, size and SHA256 are in
`data-provenance.json`; the script asserts that hash. Data executables are never
run. All reported conditioning and target times use the documented origin;
datetime targets are seconds since that object's time_origin.

The12well-conditioned configurations per source agree with independently
assembled covariance matrices and dense ordinary-kriging equations, within
1.2e-11for predictions. Four nugget-free Gaussian systems have condition number
about1.5e11; their dense-solve differences are diagnostics and excluded from that
bound. The recorded-data fits are working controls, not reproductions of the
large-time exact-surface failure. No whole-PR approval or acceptance is claimed.
