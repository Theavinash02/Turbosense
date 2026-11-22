# Notebooks

These two notebooks are the original exploration. Each is self-contained (it defines its own functions and
downloads the data), so it runs on Colab without installing the package. Outputs are saved in the files.

| Notebook | What it covers |
| --- | --- |
| `01_fd001_walkthrough.ipynb` | Data audit, labels, split by engine, the random-split comparison, baselines and the LSTM on FD001 |
| `02_all_subsets_walkthrough.ipynb` | All four subsets, per-condition scaling, and train-on-one, test-on-another |

The maintained code is the `rul` package in `src/`. It reproduces the notebooks' numbers and is what the tests cover.
