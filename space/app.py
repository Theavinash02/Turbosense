"""Gradio demo: remaining useful life and health monitoring for NASA C-MAPSS FD004 test engines."""

import random
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spaces

from rul.config import load_config
from rul.data import COLUMNS, add_condition
from rul.inference import Predictor
from rul.model import predict

ROOT = Path(__file__).parent
SUBSET = "FD004"

cfg = load_config(ROOT / "configs" / "default.yaml")
cfg["paths"]["models"] = ROOT / "models"
predictor = Predictor(cfg, SUBSET)
CAP, WINDOW = predictor.cap, predictor.window

TEST = add_condition(pd.read_csv(ROOT / "data" / f"test_{SUBSET}.txt", sep=r"\s+", header=None, names=COLUMNS))
TRUE_RUL = pd.read_csv(ROOT / "data" / f"RUL_{SUBSET}.txt", header=None)[0].to_numpy()
LENGTHS = TEST.groupby("unit").size()
ENGINES = [(f"Engine {u} ({LENGTHS[u]} flights recorded)", int(u)) for u in LENGTHS.index]
NEAR_LIMIT = 50
NEAR_FAILURE = [u for u in LENGTHS.index if TRUE_RUL[u - 1] <= NEAR_LIMIT]

# Chart colours: two series (blue, orange), one alarm colour, neutral ink and grid.
BLUE, ORANGE, ALARM = "#2a78d6", "#eb6834", "#d03b3b"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def prefix_windows(a: np.ndarray, window: int) -> np.ndarray:
    """Return one window per flight: the last `window` rows seen so far, padded with the first row."""
    padded = np.vstack([np.repeat(a[:1], window - 1, axis=0), a])
    return np.stack([padded[i : i + window] for i in range(len(a))])


@lru_cache(maxsize=64)
@spaces.GPU(duration=10)  # Hugging Face ZeroGPU needs one decorated function; the small LSTMs run on CPU
def trajectory(unit: int) -> pd.DataFrame:
    """Score one test engine at every flight of its recorded history."""
    df = TEST[TEST.unit == unit].sort_values("cycle").reset_index(drop=True)
    scaled = predictor.scaler.transform(df)
    X = prefix_windows(scaled[predictor.sensors].to_numpy(np.float32), WINDOW)
    per_seed = np.stack([predict(m, X, CAP) for m in predictor.models])
    out = pd.DataFrame({
        "flight": df["cycle"].to_numpy(),
        "pred": per_seed.mean(axis=0), "pred_min": per_seed.min(axis=0), "pred_max": per_seed.max(axis=0),
    })
    # NASA gives the true remaining life at the last recorded flight; earlier flights count back from it.
    out["true"] = TRUE_RUL[unit - 1] + (out["flight"].max() - out["flight"])
    monitor = predictor.monitor
    score = monitor.score(scaled)
    ready = out["flight"] > monitor.baseline          # the first flights define the engine's own baseline
    out["score"] = np.where(ready, score.to_numpy(), np.nan)
    out["alarm"] = monitor.alarms(scaled, score).to_numpy() & ready.to_numpy()
    return out


def base_layout(title: str, y_title: str, x_max: int) -> dict:
    axis = dict(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    return dict(
        title=dict(text=title, font=dict(color=INK, size=15), x=0.015, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK), height=340,
        margin=dict(l=55, r=20, t=50, b=45), hovermode="x unified",
        xaxis=dict(title="Flight number", range=[1, max(x_max, 2)], **axis),
        yaxis=dict(title=y_title, rangemode="tozero", **axis),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", font=dict(color=INK)),
    )


def rul_figure(t: pd.DataFrame, upto: int) -> go.Figure:
    seen = t[t.flight <= upto]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seen.flight, y=seen.pred_max, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=seen.flight, y=seen.pred_min, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.15)", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=seen.flight, y=seen.pred, mode="lines", name="Predicted", line=dict(color=BLUE, width=2),
                             hovertemplate="%{y:.0f} flights<extra>Predicted</extra>"))
    fig.add_trace(go.Scatter(x=seen.flight, y=seen.true, mode="lines", name="True (NASA)",
                             line=dict(color=ORANGE, width=2),
                             hovertemplate="%{y:.0f} flights<extra>True</extra>"))
    fig.add_hline(y=CAP, line=dict(color=MUTED, width=1), annotation_text=f"model ceiling ({CAP})",
                  annotation_font=dict(color=MUTED, size=11), annotation_position="bottom right")
    layout = base_layout("Remaining flights: predicted against true", "Flights left", int(t.flight.max()))
    layout["yaxis"]["range"] = [0, max(float(t.true.max()), CAP) * 1.08]
    fig.update_layout(**layout)
    return fig


def health_figure(t: pd.DataFrame, upto: int) -> go.Figure:
    seen = t[t.flight <= upto]
    threshold = predictor.monitor.threshold
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seen.flight, y=seen.score, mode="lines", name="Distance from own baseline",
                             line=dict(color=BLUE, width=2), showlegend=False,
                             hovertemplate="%{y:.1f}<extra>Distance</extra>"))
    fig.add_hline(y=threshold, line=dict(color=MUTED, width=1), annotation_text="alarm threshold",
                  annotation_font=dict(color=MUTED, size=11), annotation_position="top left")
    first = first_alarm(t, upto)
    if first is not None:
        y = float(t.loc[t.flight == first, "score"].iloc[0])
        fig.add_trace(go.Scatter(x=[first], y=[y], mode="markers+text", text=[f"alarm, flight {first}"],
                                 textposition="top left", textfont=dict(color=INK, size=12), showlegend=False,
                                 marker=dict(color=ALARM, size=11, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"Alarm raised at flight {first}<extra></extra>"))
    layout = base_layout("Health monitor: distance from this engine's own early flights", "Distance",
                         int(t.flight.max()))
    top = np.nanmax(t.score.to_numpy()) if t.score.notna().any() else threshold
    layout["yaxis"]["range"] = [0, max(float(top), threshold) * 1.15]
    fig.update_layout(**layout)
    return fig


def first_alarm(t: pd.DataFrame, upto: int) -> int | None:
    hits = t[(t.flight <= upto) & t.alarm]
    return int(hits.flight.iloc[0]) if len(hits) else None


def render(unit: int, upto: int):
    t = trajectory(int(unit))
    upto = int(min(max(upto, 1), t.flight.max()))
    now = t[t.flight == upto].iloc[0]
    at_cap = now.pred >= CAP - 1
    predicted = (f"### Predicted\n## {'at least ' + str(CAP) if at_cap else f'{now.pred:.0f}'} flights left\n"
                 f"Three models range from {now.pred_min:.0f} to {now.pred_max:.0f}."
                 + ("\n\nThe model cannot predict above its ceiling." if at_cap else "")
                 + (f"\n\nFewer than {WINDOW} flights seen, so the input is padded." if upto < WINDOW else ""))
    error = now.pred - now.true
    side = "too late" if error > 0 else "too early"
    truth = (f"### True answer (NASA)\n## {now.true:.0f} flights left\n"
             + (f"Above the model's ceiling of {CAP}, so no exact prediction is possible here."
                if now.true > CAP else f"The prediction is {abs(error):.0f} flights {side}."))
    monitor = predictor.monitor
    first = first_alarm(t, upto)
    if upto <= monitor.baseline:
        health = (f"### Health monitor\n## Learning baseline\nThe first {monitor.baseline} flights define "
                  "what normal looks like for this engine.")
    elif first is None:
        health = (f"### Health monitor\n## No alarm\nDistance {now.score:.1f}, threshold {monitor.threshold:.1f}. "
                  f"An alarm needs {monitor.persist} flights in a row above the threshold.")
    else:
        left = t.loc[t.flight == first, "true"].iloc[0]
        health = (f"### Health monitor\n## Alarm since flight {first}\n"
                  f"It was raised with {left:.0f} flights truly left. Distance now {now.score:.1f}, "
                  f"threshold {monitor.threshold:.1f}.")
    return predicted, truth, health, rul_figure(t, upto), health_figure(t, upto)


def choose(unit: int):
    n = int(LENGTHS[int(unit)])
    return (gr.update(maximum=n, value=n), *render(unit, n))


def choose_random():
    unit = random.choice(ENGINES)[1]
    return (gr.update(value=unit), *choose(unit))


def choose_random_near_failure():
    unit = int(random.choice(NEAR_FAILURE))
    return (gr.update(value=unit), *choose(unit))


ABOUT = f"""
**What this is.** A demo of the [Turbosense](https://github.com/Theavinash02/Turbosense) project. Each engine is a real
test engine from NASA's C-MAPSS FD004 simulation data, which the models never saw in training. Its history stops
some time before failure, and NASA publishes how many flights it had left at that point.

**How to use it.** Pick an engine, then drag the slider back to replay its life flight by flight. Many test
engines are still healthy when their record stops, so the model just reads "near the ceiling" for them. The
second button picks only from the {{n_near}} of {{n_all}} engines with {NEAR_LIMIT} flights or fewer left, where
there is more to see.

**Predicted remaining life.** Three LSTM models (PyTorch) each read the last {WINDOW} flights of 14 sensors. The
line is their average and the shaded band is their lowest and highest answer. That band is a rough spread, not
a confidence interval.

**Health monitor.** A separate, unsupervised check that uses no failure data. It measures how far the engine's
sensors have moved from its own first {predictor.monitor.baseline} flights, and raises an alarm after
{predictor.monitor.persist} flights in a row above a threshold.

**Limits.**
- The model's answers stop at {CAP} flights. A healthy engine reads as "at least {CAP}".
- Average test error is about 14 flights when the true answer is {CAP} or less, and larger above that.
- The data is simulated. Real fleets have missing sensors, maintenance events and more varied conditions.
- On held-out engines, about 1 in 10 FD004 engines got an alarm while still healthy.

Data: C-MAPSS turbofan engine degradation simulation, NASA Ames Prognostics Center of Excellence
(Saxena, Goebel, Simon and Eklund, 2008).
""".replace("{n_near}", str(len(NEAR_FAILURE))).replace("{n_all}", str(len(ENGINES)))

with gr.Blocks(title="Turbofan remaining useful life") as demo:
    gr.Markdown("# Turbofan engine: remaining useful life and health monitor\n"
                "How many flights does a jet engine have left before it fails? Pick an engine the models have "
                "never seen and replay its sensor history.")
    start = random.choice(ENGINES)[1]
    with gr.Row():
        engine = gr.Dropdown(choices=ENGINES, value=start, label="Test engine", scale=3)
        shuffle = gr.Button("Random engine", scale=1)
        shuffle_near = gr.Button("Random engine near failure", scale=1)
    flights = gr.Slider(1, int(LENGTHS[start]), value=int(LENGTHS[start]), step=1,
                        label="Flights observed so far (drag back to replay the engine's life)")
    with gr.Row():
        predicted, truth, health = gr.Markdown(), gr.Markdown(), gr.Markdown()
    rul_plot, health_plot = gr.Plot(show_label=False), gr.Plot(show_label=False)
    with gr.Accordion("About this demo", open=False):
        gr.Markdown(ABOUT)

    outputs = [predicted, truth, health, rul_plot, health_plot]
    engine.input(choose, engine, [flights, *outputs])
    shuffle.click(choose_random, None, [engine, flights, *outputs])
    shuffle_near.click(choose_random_near_failure, None, [engine, flights, *outputs])
    flights.release(render, [engine, flights], outputs)
    demo.load(render, [engine, flights], outputs)

if __name__ == "__main__":
    demo.launch()
