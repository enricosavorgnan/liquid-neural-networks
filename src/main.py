import argparse
import numpy as np
import torch
import yaml

from src.data import *
from src.models import *
from src.models.cfc import *
from src.models.wccfc import *
from src.train import *
from src.utils import *
from src.vis import *



def main(config):
    # Define variables

    # Simulate data
    train, validation, test = get_data()

    # Define model
    model = get_model()

    # Train model
    train_performance = train(model=model, train_loader=train, val_loader=validation)

    # Evaluate model
    eval_performance = evaluate(model=model, test_loader=test)

    # Plot results
    figs, names = plot(model, train_performance, eval_performance)

    # Save information
    save(model=model, figs=figs, names=names, performance=[train_performance, eval_performance])

    return



if __name__ == "__main__":
    config = {
        "mode": "train",                 # [ train | eval ]
        "plot": True,
        "data": {
          "name": None,
        },
        "model": {
            "name": CfC
        },
        "save": {
            "save": True,
            "path": "."
        }
    }

    parser = argparse.ArgumentParser(description="Closed-form Continuous-Time Neural Networks \n "
                                                 "Add `--config [path/to/config.yaml]` to load a YAML config file")
    parser.add_argument("--config", type=str, help="Path to the config file (YAML format).")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = parse_config(config)

    main(config)
