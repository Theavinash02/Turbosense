# Turbofan remaining useful life

Predicts how many flight cycles a jet engine has left before failure, from its sensor history, on NASA's
C-MAPSS turbofan dataset. A PyTorch LSTM is compared with a LightGBM baseline on all four subsets, with runs
tracked in MLflow. An unsupervised health monitor flags engines that have left their healthy regime, and a
FastAPI service, packaged with Docker, returns both for a given engine history.

**Short version**

- Split by engine, the LSTM reaches a test error of about 14.5 cycles on every subset, including the two with
  six operating conditions, and beats LightGBM on each.
- A random row split would have reported 9.75 cycles for a model that scores 15.83 on unseen engines
  (notebook 1).
- Standardising sensors within each operating condition is what makes the harder subsets work: on FD004 the
  LSTM's error falls from 19.28 to 14.52.
- A model trained on the broadest subset (FD004) works on all four test sets. Models trained on a narrower
  subset fail on conditions or fault modes they never saw, often no better than a constant guess.
- The health monitor, which uses no failure labels, raised an alarm on every held-out engine before it failed,
  a median of 64.5 to 76.5 cycles ahead depending on the subset. Between 0% and 10% of engines got an alarm while
  still in the healthy range.

## Results

Test error in flight cycles, one prediction per held-out test engine at its last observed cycle. Full tables
are in [reports/results.md](reports/results.md), which `rul report` writes from the JSON files next to it.

| Subset | Conditions / fault modes | Test engines | LightGBM RMSE | LSTM RMSE | LSTM RMSE, true RUL |
| --- | --- | --- | --- | --- | --- |
| FD001 | 1 / 1 | 100 | 15.88 | 14.40 ± 0.23 | 15.79 |
| FD002 | 6 / 1 | 259 | 15.78 | 14.29 ± 0.26 | 26.76 |
| FD003 | 1 / 2 | 100 | 17.37 | 14.65 ± 0.59 | 16.05 |
| FD004 | 6 / 2 | 248 | 18.01 | 14.52 ± 0.44 | 26.95 |

RMSE is against the true RUL capped at 125 cycles unless marked. LSTM figures are the mean over 3 seeds,
± one standard deviation. On FD002 and FD004 the error against the uncapped truth is much larger, because
many test engines have more than 125 cycles left and the model cannot predict above the cap.

Train on one subset (row), test on another (column), LSTM, no retraining:

| Trained on | FD001 | FD002 | FD003 | FD004 |
| --- | --- | --- | --- | --- |
| FD001 | 14.4 | 42.4 | 39.9 | 46.2 |
| FD002 | 15.5 | 14.3 | 34.2 | 30.9 |
| FD003 | 15.2 | 49.0 | 14.6 | 47.9 |
| FD004 | 16.4 | 14.8 | 14.3 | 14.5 |

For scale, always predicting the training mean scores between 42 and 46 on these test sets.

### Health monitor

Measured on held-out validation engines that run to failure. An early alarm is one that fires with 125 or
more cycles left. AUROC is measured on the test engines at their last observed cycle: how well the score
separates engines within 30 cycles of failure from the rest.

| Subset | Reference | Engines alarmed before failure | Median lead (cycles) | Early alarms | AUROC, RUL ≤ 30 |
| --- | --- | --- | --- | --- | --- |
| FD001 | own baseline | 100% | 76.5 | 5% | 0.967 |
| FD001 | fleet mean | 50% | 28.0 | 0% | 0.932 |
| FD002 | own baseline | 100% | 64.5 | 2% | 0.985 |
| FD002 | fleet mean | 46% | 16.0 | 0% | 0.950 |
| FD003 | own baseline | 100% | 72.0 | 0% | 0.974 |
| FD003 | fleet mean | 90% | 21.0 | 0% | 0.952 |
| FD004 | own baseline | 100% | 70.0 | 10% | 0.965 |
| FD004 | fleet mean | 76% | 24.0 | 4% | 0.928 |

Comparing each engine with its own first 20 cycles catches every engine. Comparing it with the fleet's healthy
mean misses up to half of them, because normal engine-to-engine differences are as large as early degradation.

## Design choices

- **Split by engine.** Consecutive cycles of one engine are nearly identical, so a random row split tests the
  model on engines it has already seen. Whole engines are held out for validation, and `check_split` fails
  the run if an engine appears on both sides.
- **Capped target.** Early in life the sensors look healthy and carry no information about whether failure is
  150 or 250 cycles away. The target is clipped at 125 cycles, a common choice for this dataset. Both capped
  and uncapped test errors are reported.
- **Per-condition scaling.** In FD002 and FD004 raw sensor values mostly reflect the flight regime. Each sensor
  is standardised within its operating condition, using statistics from the training engines only. A condition
  that was never seen falls back to the global statistics.
- **Two metrics.** RMSE, and the NASA score from the PHM08 challenge, which penalises a late prediction more
  than an early one.
- **Several seeds.** Each LSTM is trained three times; the spread is reported next to the mean.
- **Health monitor separate from the RUL model.** It is the Mahalanobis distance of an engine's smoothed sensors
  from its own early-life baseline, fitted on healthy cycles only, so it needs no failure history. An alarm
  needs five cycles in a row above the threshold, so a single bad reading does not trigger it.
- **FD004 models are served.** They were trained on all six operating conditions and both fault modes, and
  the transfer table shows they hold up on every subset.

## Run it

Needs Python 3.10+ and git. Everything runs on CPU; a GPU is used if present.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[train,api,dev]"
rul all                          # download, train everything, fit monitors, transfer, report
```

`rul all` took about 16 minutes on a 2-core CPU. Individual steps:

```bash
rul download
rul train --subset FD001                      # LightGBM baseline + LSTM (3 seeds)
rul train --subset FD002 --scaling global     # the scaling comparison
rul anomaly --subset FD001                    # fit and evaluate the health monitor
rul transfer                                  # needs all four subsets trained
rul report
```

Settings are in [configs/default.yaml](configs/default.yaml).

```bash
pytest -q        # unit tests plus end-to-end runs on synthetic engines; no download needed
ruff check src tests
```

## Prediction API

The trained FD004 models are committed in `models/FD004_condition/` (under 1 MB), so the API runs without
training anything first.

```bash
pip install -e ".[api]"
rul serve                        # http://127.0.0.1:8000, interactive docs at /docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/request.json
```

`examples/request.json` is the full history of one real FD004 test engine (108 cycles; NASA's true answer is
25 cycles left). The service returns:

```json
{
  "engine_id": "FD004-test-77",
  "model_subset": "FD004",
  "cycles_received": 108,
  "padded": false,
  "rul_cycles": 34.5,
  "rul_min": 31.2,
  "rul_max": 38.1,
  "at_cap": false,
  "unseen_condition_share": 0.0,
  "anomaly": {"available": true, "score": 10.83, "threshold": 6.25, "alarm": true}
}
```

(Numbers rounded here; [examples/response.json](examples/response.json) has the exact output.) The engine was
picked for its short history, not for a flattering result: the model is 9.5 cycles late on it.

- `rul_min` and `rul_max` are the lowest and highest of the three seeds, a rough spread and not a confidence
  interval.
- `at_cap` is true when the prediction sits at the 125-cycle ceiling: read it as "at least 125".
- `unseen_condition_share` above zero means part of the history is in an operating condition the model was
  not trained on, and the prediction should not be trusted.
- The anomaly result needs the history from the engine's first cycle and more than 20 cycles; otherwise it is
  returned as unavailable with the reason.
- Bad input (no cycles, a missing sensor, cycles out of order) is rejected with HTTP 422.

### Demo app

`space/` is a Gradio app for Hugging Face Spaces. Pick a FD004 test engine and replay its sensor history: the
predicted flights left next to NASA's true answer, and the health monitor's alarm. It is self-contained: `space/rul/`
is a copy of the inference modules from `src/rul/`, with the FD004 models and test data beside it.

```bash
cd space
pip install -r requirements.txt gradio
python app.py                    # http://127.0.0.1:7860
```

To deploy, create a Gradio Space and upload the contents of `space/`. A change to `src/rul/` must be copied into
`space/rul/` by hand.

### Docker

```bash
docker build -t turbofan-rul .
docker run --rm -p 8000:8000 turbofan-rul
```

The image installs CPU-only PyTorch and only the serving dependencies (no MLflow or LightGBM), and runs as a
non-root user. The CI workflow builds it and sends the example request on every push.

## Experiment tracking

Every training run is logged to MLflow: parameters, validation and test metrics, per-epoch loss curves, and
the saved model and scaler. The store is a local SQLite file.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Pass `--no-mlflow` to `rul train` or `rul all` to skip logging.

## Layout

```
src/rul/      data, scaling, windows, model, train, evaluate, transfer, anomaly, inference, api, tracking,
              report, cli
configs/      every setting and seed
models/       FD004 weights, scaler and health monitor for serving (other subsets are not committed)
reports/      metrics JSON per run and results.md
examples/     a sample API request and its response
notebooks/    the two exploration notebooks, with outputs
tests/        unit tests, API tests and end-to-end runs on synthetic data
space/        the Gradio demo app for Hugging Face Spaces
Dockerfile    the serving image
```

## Limitations

- C-MAPSS is simulated. Real fleets have missing sensors, maintenance events and continuously varying
  operating conditions.
- Operating conditions are found by rounding the settings columns, which works only because the simulator
  uses six fixed regimes.
- Predictions cannot exceed 125 cycles, so healthy engines cannot be ranked against each other.
- The model gives a point estimate with no uncertainty interval.
- Three seeds and no per-subset tuning: differences of about a cycle between models are within noise.
- Why the FD001 model fails on FD003 (same operating condition, one extra fault mode) has not been
  investigated.
- The health monitor's settings (smoothing, baseline length, persistence) were chosen once and not tuned, and
  its alarm statistics come from 20 to 52 validation engines per subset. On FD004, 10% of engines got an alarm
  while still healthy.
- The monitor needs each engine's history from its first cycle. An engine that enters monitoring mid-life has
  no baseline.
- The API has no authentication, rate limiting or request logging. It is a demonstration service.
- Results have not been compared against published numbers in this repository.

## Data

C-MAPSS turbofan engine degradation simulation data, NASA Ames Prognostics Center of Excellence.
A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation", PHM08, 2008.

`rul download` fetches the files from a GitHub mirror (`edwardzjl/CMAPSSData`), which states no licence of
its own. Data files are not committed to this repository.

## License

MIT, see [LICENSE](LICENSE).
