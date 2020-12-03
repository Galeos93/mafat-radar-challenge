import os
from pathlib import Path
import imgaug
from functools import partial
import random
import re
from typing import Any, List, Tuple, Dict
from types import ModuleType
import yaml

from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
from torchsummary import summary

import mafat_radar_challenge.data_loader.augmentation as module_aug
import mafat_radar_challenge.data_loader.mixers as module_mix
import mafat_radar_challenge.data_loader.data_loaders as module_data
import mafat_radar_challenge.model.loss as module_loss
import mafat_radar_challenge.model.metric as module_metric
import mafat_radar_challenge.model.model as module_arch
from mafat_radar_challenge.trainer import Trainer, MAFATTrainer
from mafat_radar_challenge.utils import (
    setup_logger,
    trainer_paths,
    moving_average,
    bn_update,
)
from mafat_radar_challenge.tester import MAFATTester
import mafat_radar_challenge.data_loader.data_splitter as module_splitter
import mafat_radar_challenge.data_loader.samplers as module_sampler


log = setup_logger(__name__)


def train_k_fold(cfg: Dict, resume: str) -> None:
    for fold in range(cfg["folds"]):
        print("Running fold {}".format(fold))

        cfg["name"] = cfg["name"].format(fold)
        cfg["data_loader"]["args"]["data_dir"] = cfg["data_loader"]["args"][
            "data_dir"
        ].format(fold)
        cfg["data_loader"]["args"]["csv_dir"] = cfg["data_loader"]["args"][
            "csv_dir"
        ].format(fold)
        cfg["val_data_loader"]["args"]["data_dir"] = cfg["data_loader"]["args"][
            "data_dir"
        ].format(fold)
        cfg["val_data_loader"]["args"]["csv_dir"] = cfg["data_loader"]["args"][
            "csv_dir"
        ].format(fold)
        print(cfg["name"])

        train(cfg, resume)


def train(cfg: Dict, resume: str) -> None:
    log.debug(f"Training: {cfg}")
    seed_everything(cfg["seed"])

    model = get_instance(module_arch, "arch", cfg)
    model, device = setup_device(model, cfg["target_devices"])
    if cfg["data_loader"]["args"]["use_metadata"]:
        pass
        # summary(model, [(3, 126, 32), (7,)])
    else:
        summary(model, [3, 64, 32])
    torch.backends.cudnn.deterministic = True  # Enables determinism
    torch.backends.cudnn.benchmark = (
        False  # enable if not consistent input sizes and undeterministic
    )

    param_groups = setup_param_groups(model, cfg["optimizer"])
    optimizer = get_instance(module_optimizer, "optimizer", cfg, param_groups)
    lr_scheduler = get_instance(module_scheduler, "lr_scheduler", cfg, optimizer)
    model, optimizer, start_epoch = resume_checkpoint(resume, model, optimizer, cfg)

    transforms = get_instance(module_aug, "augmentation", cfg)

    if "mixer" in cfg:
        mixer = get_instance(module_mix, "mixer", cfg)
    else:
        mixer = None

    # Sampler addition
    if "sampler" in cfg:
        sampler = getattr(module_sampler, cfg["sampler"]["type"])
        sampler = partial(sampler, **cfg["sampler"]["args"])
    else:
        sampler = None

    data_loader = get_instance(
        module_data, "data_loader", cfg, transforms, sampler, mixer
    )
    valid_data_loader = get_instance(
        module_data, "val_data_loader", cfg, transforms, sampler
    )

    log.info("Getting loss and metric function handles")

    if isinstance(cfg["loss"], dict):
        loss = getattr(module_loss, cfg["loss"]["type"])
        loss = partial(loss, **cfg["loss"]["args"])
    else:
        loss = getattr(module_loss, cfg["loss"])
    metrics = [getattr(module_metric, met) for met in cfg["metrics"]]

    log.info("Initialising trainer")
    trainer = MAFATTrainer(
        model,
        loss,
        metrics,
        optimizer,
        start_epoch=start_epoch,
        config=cfg,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()
    log.info("Finished!")


def test(cfg):
    log.debug(f"Testing: {cfg}")
    seed_everything(cfg["seed"])

    torch.backends.cudnn.deterministic = True  # Enables determinism
    torch.backends.cudnn.benchmark = (
        False  # enable if not consistent input sizes and undeterministic
    )

    transforms = get_instance(module_aug, "augmentation", cfg)
    data_loader = get_instance(module_data, "data_loader", cfg, transforms)
    model = get_instance(module_arch, "arch", cfg)

    model_checkpoint = cfg["model_checkpoint"]
    log.debug(f"Loading checkpoint {model_checkpoint}")
    checkpoint = torch.load(model_checkpoint)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    model, device = setup_device(model, cfg["target_devices"])
    model.eval()

    log.info("Testing...")
    tester = MAFATTester(cfg, device, data_loader)
    submission_df = tester.create_submission(model, data_loader)
    output_csv = os.path.join(
        cfg["submission_dir"], "submission-{}.csv".format(cfg["name"])
    )
    submission_df.to_csv(output_csv, index=False)
    # Download zip file

    with ZipFile(os.path.splitext(output_csv)[0] + ".zip", "w") as myzip:
        myzip.write(output_csv, arcname=os.path.basename(output_csv))

    return submission_df


def test_tta(cfg):
    log.debug(f"Testing: {cfg}")
    seed_everything(cfg["seed"])

    transforms = get_instance(module_aug, "augmentation", cfg)
    transforms_list = getattr(transforms, "TRANSFORM_LIST")

    model = get_instance(module_arch, "arch", cfg)

    model_checkpoint = cfg["model_checkpoint"]
    log.debug(f"Loading checkpoint {model_checkpoint}")
    checkpoint = torch.load(model_checkpoint)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    model, device = setup_device(model, cfg["target_devices"])
    model.eval()

    log.info("Testing...")
    targets = list()
    for idx, transformation in enumerate(transforms_list):
        log.info("Transformation {}".format(idx))
        transforms.CURR_TRANSFORM = transformation
        data_loader = get_instance(module_data, "data_loader", cfg, transforms)
        tester = MAFATTester(cfg, device, data_loader)
        submission_df = tester.create_submission(model, data_loader)
        targets.append(np.array(submission_df.prediction))
    submission_df.prediction = np.mean(targets, axis=0)

    output_csv = os.path.join(
        cfg["submission_dir"], "submission-{}-TTA.csv".format(cfg["name"])
    )

    submission_df.to_csv(output_csv, index=False)
    with ZipFile(os.path.splitext(output_csv)[0] + ".zip", "w") as myzip:
        myzip.write(output_csv, arcname=os.path.basename(output_csv))


def apply_swa(checkpoint_path, epochs=10, save=True):
    with open(os.path.join(os.path.dirname(checkpoint_path), "config.yml")) as fh:
        cfg = yaml.safe_load(fh)
    log.debug(f"Training: {cfg}")
    seed_everything(cfg["seed"])
    model = get_instance(module_arch, "arch", cfg)
    model, device = setup_device(model, cfg["target_devices"])
    swa_model, device = setup_device(model, cfg["target_devices"])
    torch.backends.cudnn.deterministic = True  # Enables determinism
    torch.backends.cudnn.benchmark = (
        False  # enable if not consistent input sizes and undeterministic
    )

    param_groups = setup_param_groups(model, cfg["optimizer"])
    optimizer = get_instance(module_optimizer, "optimizer", cfg, param_groups)

    transforms = get_instance(module_aug, "augmentation", cfg)

    if "mixer" in cfg:
        mixer = get_instance(module_mix, "mixer", cfg)
    else:
        mixer = None

    # Sampler addition
    if "sampler" in cfg:
        sampler = getattr(module_sampler, cfg["sampler"]["type"])
        sampler = partial(sampler, **cfg["sampler"]["args"])
    else:
        sampler = None

    data_loader = get_instance(
        module_data, "data_loader", cfg, transforms, sampler, mixer
    )
    for swa_n in range(epochs):
        model, optimizer, start_epoch = resume_checkpoint(
            checkpoint_path, model, optimizer, cfg
        )
        if swa_n == 0:
            start = start_epoch
        print("Start epoch: ", start_epoch)
        moving_average(swa_model, model, 1.0 / (swa_n + 1))
        if swa_n == epochs - 1:
            bn_update(data_loader, swa_model)
        checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "checkpoint-epoch{}.pth".format(start_epoch + 1),
        )
        end = start_epoch
        print("Finished")
        if not os.path.isfile(checkpoint_path):
            break

    ensemble_path = os.path.join(
        os.path.dirname(checkpoint_path),
        "checkpoint-epoch{}-{}.pth".format(start, end),
    )
    if save:
        print("Saving ensemble...")
        torch.save({"state_dict": swa_model.state_dict()}, ensemble_path)
    return swa_model, ensemble_path


def setup_device(
    model: nn.Module, target_devices: List[int]
) -> Tuple[torch.device, List[int]]:
    """
    setup GPU device if available, move model into configured device
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        log.warning(
            "There's no GPU available on this machine. Training will be performed on CPU."
        )
        device = torch.device("cpu")
        model = model.to(device)
        return model, device

    if not target_devices:
        log.info("No GPU selected. Training will be performed on CPU.")
        device = torch.device("cpu")
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (
            f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu} "
            "available. Check the configuration and try again."
        )
        log.critical(msg)
        raise Exception(msg)

    log.info(f"Using devices {target_devices} of available devices {available_devices}")
    device = torch.device(f"cuda:{target_devices[0]}")
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices)
    else:
        model = model.to(device)
    return model, device


def setup_param_groups(model: nn.Module, config: Dict) -> List:
    return [{"params": model.parameters(), **config}]


def resume_checkpoint(resume_path, model, optimizer, config):
    """
    Resume from saved checkpoint.
    """
    if not resume_path:
        return model, optimizer, 0

    log.info(f"Loading checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["state_dict"])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint["config"]["optimizer"]["type"] != config["optimizer"]["type"]:
        log.warning(
            "Warning: Optimizer type given in config file is different from "
            "that of checkpoint. Optimizer parameters not being resumed."
        )
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])

    log.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, checkpoint["epoch"]


def get_instance(module: ModuleType, name: str, config: Dict, *args: Any) -> Any:
    """
    Helper to construct an instance of a class.

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    name : str
        Name of class, as would be returned by ``.__class__.__name__``.
    config : dict
        Dictionary containing an 'args' item, which will be used as ``kwargs`` to construct the
        class instance.
    args : Any
        Positional arguments to be given before ``kwargs`` in ``config``.
    """
    ctor_name = config[name]["type"]
    module_name = getattr(module, "__name__", str(module))
    log.info(f"Building: {module_name}.{ctor_name}")
    return getattr(module, ctor_name)(*args, **config[name]["args"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    imgaug.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
