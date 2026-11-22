import argparse

from rul.config import load_config

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rul", description="Remaining useful life on NASA C-MAPSS turbofan data.")
    parser.add_argument("--config", default=None, help="YAML settings file (default: configs/default.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch the C-MAPSS files")
    p_train = sub.add_parser("train", help="train the baseline and the LSTM on one subset")
    p_train.add_argument("--subset", required=True, choices=SUBSETS)
    p_train.add_argument("--scaling", default="condition", choices=["condition", "global"])
    p_train.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p_anomaly = sub.add_parser("anomaly", help="fit and evaluate the health monitor on one subset")
    p_anomaly.add_argument("--subset", required=True, choices=SUBSETS)
    sub.add_parser("transfer", help="apply each subset's models to every subset's test engines")
    sub.add_parser("report", help="write reports/results.md from the JSON reports")
    p_all = sub.add_parser("all", help="download, train everything, fit monitors, transfer and report")
    p_all.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p_serve = sub.add_parser("serve", help="start the prediction API")
    p_serve.add_argument("--subset", default=None, choices=SUBSETS, help="which saved models to serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    # Imports are per command so that `rul serve` works without the training extras installed.
    if args.command == "download":
        from rul import data
        print("data in", data.download(cfg))
    elif args.command == "train":
        from rul import train
        train.train_subset(cfg, args.subset, args.scaling, track=not args.no_mlflow)
    elif args.command == "anomaly":
        from rul import anomaly
        anomaly.run_anomaly(cfg, args.subset)
    elif args.command == "transfer":
        from rul import transfer
        transfer.transfer_matrix(cfg)
    elif args.command == "report":
        from rul import report
        print(report.build_report(cfg))
    elif args.command == "all":
        from rul import anomaly, data, report, train, transfer
        data.download(cfg)
        for subset in cfg["data"]["subsets"]:
            train.train_subset(cfg, subset, "condition", track=not args.no_mlflow)
            anomaly.run_anomaly(cfg, subset)
        for subset in ("FD002", "FD004"):  # the two subsets with several operating conditions
            if subset in cfg["data"]["subsets"]:
                train.train_subset(cfg, subset, "global", track=not args.no_mlflow)
        transfer.transfer_matrix(cfg)
        print(report.build_report(cfg))
    elif args.command == "serve":
        import uvicorn

        from rul.api import create_app
        uvicorn.run(create_app(cfg, args.subset), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
