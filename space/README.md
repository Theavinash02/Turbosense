---
title: Turbofan Remaining Useful Life
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predict flights left before a jet engine fails
---

# Turbofan remaining useful life

Pick a NASA C-MAPSS test engine and replay its sensor history. The demo shows:

- **Predicted remaining flights** from three PyTorch LSTM models, next to NASA's true answer.
- **A health monitor**: an unsupervised alarm that fires when the engine drifts from its own early flights.

The models were trained on the FD004 training engines only; every engine here is from the held-out test set.

## Files

| Path | What it is |
| --- | --- |
| `app.py` | The Gradio app |
| `rul/` | Inference code copied from the main project (scaler, model, health monitor) |
| `models/FD004_condition/` | Three LSTM weights, the scaler and the health monitor |
| `data/` | FD004 test engines and their true remaining life (NASA C-MAPSS) |
| `configs/default.yaml` | Model settings |

## Limits

- Predictions stop at 125 flights; a healthy engine reads as "at least 125".
- The data is simulated, not from a real fleet.
- The spread across the three models is not a confidence interval.

## Data

C-MAPSS turbofan engine degradation simulation data, NASA Ames Prognostics Center of Excellence.
A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation", PHM08, 2008.
