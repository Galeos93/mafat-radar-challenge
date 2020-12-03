import click
import copy
import numpy as np
import os
import yaml
from zipfile import ZipFile

from mafat_radar_challenge import main
from mafat_radar_challenge.utils import setup_logging


@click.group()
def cli():
    """
    CLI for mafat_radar_challenge
    """
    pass


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
@click.option("-r", "--resume", default=None, type=str, help="path to checkpoint")
def train(config_filename, resume):
    """
    Entry point to start training run(s).
    """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        setup_logging(config)
        main.train(config, resume)


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
@click.option("-r", "--resume", default=None, type=str, help="path to checkpoint")
def train_inter_location(config_filename, resume):
    """
    Entry point to start training run(s).
    """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        setup_logging(config)
        main.train_inter_location(config, resume)


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
@click.option("-r", "--resume", default=None, type=str, help="path to checkpoint")
def train_k_fold(config_filename, resume):
    """
    Entry point to start training run(s).
    """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        config_template = copy.deepcopy(config)
        for fold in range(config["folds"]):
            print("Running fold {}".format(fold))
            config["name"] = config_template["name"].format(fold)
            config["data_loader"]["args"]["data_dir"] = config_template["data_loader"][
                "args"
            ]["data_dir"].format(fold)
            config["data_loader"]["args"]["csv_dir"] = config_template["data_loader"][
                "args"
            ]["csv_dir"].format(fold)
            config["val_data_loader"]["args"]["data_dir"] = config_template[
                "val_data_loader"
            ]["args"]["data_dir"].format(fold)
            config["val_data_loader"]["args"]["csv_dir"] = config_template[
                "val_data_loader"
            ]["args"]["csv_dir"].format(fold)
            print(config["name"])
            setup_logging(config)
            main.train(config, resume)


def load_config(filename: str) -> dict:
    """
    Load a configuration file as YAML.
    """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/evaluation/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
def evaluate(config_filename):
    """
    Entry point to start training run(s).
    """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        # setup_logging(config)
        main.test(config)


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/evaluation/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
def evaluate_k_fold(config_filename):
    """
    Entry point to start training run(s).
    """
    predictions = list()
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        config_template = copy.deepcopy(config)
        for fold, model_checkpoint in enumerate(config["model_checkpoints"]):
            print("Running fold {}".format(fold))
            config["name"] = config_template["name"].format(fold)
            config["model_checkpoint"] = model_checkpoint
            print(config["name"])
            print(config["model_checkpoint"])
            # setup_logging(config)
            submission_df = main.test(config)
            predictions.append(submission_df.prediction)
    submission_df.prediction = np.mean(predictions, axis=0)

    output_csv = os.path.join(
        config_template["submission_dir"],
        "submission-{}-CV.csv".format(config_template["name"].format("MERGE")),
    )

    submission_df.to_csv(output_csv, index=False)
    with ZipFile(os.path.splitext(output_csv)[0] + ".zip", "w") as myzip:
        myzip.write(output_csv, arcname=os.path.basename(output_csv))


@cli.command()
@click.option(
    "-c",
    "--config-filename",
    default=["experiments/evaluation/config.yml"],
    multiple=True,
    help=(
        "Path to training configuration file. If multiple are provided, runs will be "
        "executed in order"
    ),
)
def evaluate_tta(config_filename):
    """
    Entry point to start training run(s).
    """
    configs = [load_config(f) for f in config_filename]
    for config in configs:
        # setup_logging(config)
        main.test_tta(config)


@cli.command()
@click.option("-r", "--resume", default=None, type=str, help="path to checkpoint")
@click.option(
    "--n_epochs",
    default=10,
    type=int,
    help="Epochs to averagem, starting from checkpoint",
)
def create_swa(resume, n_epochs):
    """
    Entry point to average model checkpoints.
    """
    main.apply_swa(resume, n_epochs)
