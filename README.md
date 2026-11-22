# Turbosense: turbofan remaining useful life

[![ci](https://github.com/Theavinash02/Turbosense/actions/workflows/ci.yml/badge.svg)](https://github.com/Theavinash02/Turbosense/actions/workflows/ci.yml)
[![Demo on Hugging Face](https://img.shields.io/badge/demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/theavinash02/turbosense)

Predicts how many flight cycles a jet engine has left before failure, from its sensor history, on NASA's
C-MAPSS dataset. It also raises an early-warning alarm when an engine starts to drift from its healthy state,
without using any failure labels.

**Live demo:** [huggingface.co/spaces/theavinash02/turbosense](https://huggingface.co/spaces/theavinash02/turbosense)

**Stack:** PyTorch (LSTM), LightGBM, pandas, MLflow, FastAPI, Docker, GitHub Actions, Gradio.

## Highlights

- **About 14.5 cycles RMSE on all four subsets.** The LSTM beats the LightGBM baseline on each one,
  including the two with six operating conditions.
- **No data leakage.** Data is split by engine, not by row. A random row split would have reported 9.75 cycles for
  a model that really scores 15.83 on unseen engines.
- **Per-condition scaling.** Standardising sensors within each operating condition cuts FD004 error
  from 19.28 to 14.52.
- **Unsupervised health monitor.** It raised an alarm on every held-out engine before failure, a median of
  64.5 to 76.5 cycles ahead.
- **Production pieces.** MLflow tracking, a FastAPI service in Docker, 27 tests and CI on every push.

## Results

LSTM: mean of 3 seeds ± one standard deviation. RMSE is against true RUL capped at 125 cycles. Full tables are
in [reports/results.md](reports/results.md).

| Subset | Conditions / fault modes | Test engines | LightGBM RMSE | LSTM RMSE |
| --- | --- | --- | --- | --- |
| FD001 | 1 / 1 | 100 | 15.88 | 14.40 ± 0.23 |
| FD002 | 6 / 1 | 259 | 15.78 | 14.29 ± 0.26 |
| FD003 | 1 / 2 | 100 | 17.37 | 14.65 ± 0.59 |
| FD004 | 6 / 2 | 248 | 18.01 | 14.52 ± 0.44 |

**Transfer.** A model trained on FD004, the broadest subset, scores 14.3 to 16.4 on all four test sets. Models
trained on narrower subsets fail on conditions they never saw: for example, the FD001 model scores 46.2 on FD004.

**Health monitor.** This compares each engine's sensors with its own first 20 cycles, using the Mahalanobis
distance. It caught 100% of validation engines before failure, and 0% to 10% of them got an alarm while still
healthy. A fleet-average baseline caught only 46% to 90%.

## How it works

1. **Labels:** remaining cycles to failure, capped at 125, because early sensor readings carry no information
   about distant failure.
2. **Split:** whole engines are held out for validation, and a check fails the run if any engine appears on
   both sides.
3. **Scaling:** each sensor is standardised within its operating condition, using training engines only.
4. **Features:** 30-cycle sliding windows of 14 sensors.
5. **Models:** a LightGBM baseline on rolling statistics, and a 2-layer LSTM with early stopping. Both are
   scored by RMSE and the NASA score, which penalises late predictions more than early ones.
6. **Monitor:** an alarm fires after five cycles in a row above a threshold fitted on healthy cycles.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[train,api,dev]"
rul all          # download data, train all subsets, fit monitors, transfer test, report (~16 min on CPU)
pytest -q        # tests run on synthetic engines, no download needed
```

Settings are in [configs/default.yaml](configs/default.yaml). Runs are logged to MLflow; open the UI with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Prediction API

The trained FD004 models are committed, so the API runs without training first.

```bash
rul serve        # http://127.0.0.1:8000, interactive docs at /docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/request.json
```

The response gives the predicted cycles left, the spread across the 3 seeds, and the health-monitor score
and alarm. See [examples/response.json](examples/response.json) for the full output.

With Docker:

```bash
docker build -t turbofan-rul .
docker run --rm -p 8000:8000 turbofan-rul
```

## Demo app

`space/` is the self-contained Gradio app behind the live demo. To run it locally:

```bash
cd space && pip install -r requirements.txt gradio && python app.py   # http://127.0.0.1:7860
```

## Project layout

```
src/rul/      data, scaling, windows, model, train, evaluate, anomaly, inference, api, cli
configs/      all settings and seeds
models/       FD004 weights, scaler and health monitor used for serving
reports/      metrics per run and results.md
notebooks/    exploration notebooks with outputs
tests/        unit, API and end-to-end tests
space/        Gradio demo app for Hugging Face Spaces
```

## Limitations

- C-MAPSS is simulated: real fleets have missing sensors, maintenance events and changing conditions.
- Predictions stop at 125 cycles, and there is no uncertainty interval.
- The monitor's settings were not tuned, and it needs each engine's history from its first cycle.

## Data and license

Data: NASA C-MAPSS turbofan degradation simulation (Saxena et al., PHM08, 2008). It is downloaded by
`rul download` and is not committed to this repository. Code: MIT, see [LICENSE](LICENSE).
