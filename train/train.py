import os
import sys
import argparse
# Add project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import pandas as pd
from trainer.trainer import Trainer
from trainer.io import setup, set_seeds
from dataset.audiodata import SupervisedAudioData
from network.autoencoder.autoencoder import AutoEncoder
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam

if __name__ == "__main__":
    import torch.multiprocessing as mp
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--num_step', type=int, default=100000, help='Number of training steps')

    args = parser.parse_args()

    # ------------------------------- Config Loading ------------------------------- #
    config = setup(default_config="../configs/sax.yaml")
    config = OmegaConf.create(vars(config)) if not isinstance(config, OmegaConf) else config
    print(OmegaConf.to_yaml(config))

    # ------------------------------- Setup ------------------------------- #
    set_seeds(config.seed)
    Trainer.set_experiment_name(config.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------- Model & Loss ------------------------------- #
    net = AutoEncoder(config).to(device)
    loss_fn = MSSLoss([2048, 1024, 512, 256], use_reverb=config.use_reverb).to(device)

    # ------------------------------- Metric ------------------------------- #
    if config.metric == "mss":
        def metric(output, gt):
            with torch.no_grad():
                return -loss_fn(output, gt)
    elif config.metric == "f0":
        raise NotImplementedError("f0 metric not implemented yet.")
    else:
        raise NotImplementedError(f"Unknown metric: {config.metric}")

    # ------------------------------- Dataset ------------------------------- #
    def prepare_csv_paths(data_paths, frame_res):
        return [
            os.path.join(
                os.path.dirname(wav),
                f"f0_{frame_res:.3f}",
                os.path.basename(wav).replace(".wav", ".f0.csv")
            ) for wav in data_paths
        ]

    train_data = glob.glob(os.path.join(config.train, "*.wav")) * config.batch_size
    valid_data = glob.glob(os.path.join(config.test, "*.wav"))

    train_dataset = SupervisedAudioData(
        sample_rate=config.sample_rate,
        paths=train_data,
        csv_paths=prepare_csv_paths(train_data, config.frame_resolution),
        seed=config.seed,
        waveform_sec=config.waveform_sec,
        frame_resolution=config.frame_resolution,
    )

    valid_dataset = SupervisedAudioData(
        sample_rate=config.sample_rate,
        paths=valid_data,
        csv_paths=prepare_csv_paths(valid_data, config.frame_resolution),
        seed=config.seed,
        waveform_sec=config.valid_waveform_sec,
        frame_resolution=config.frame_resolution,
        random_sample=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.batch_size // (config.valid_waveform_sec / config.waveform_sec)),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    # ------------------------------- Optimizer & Scheduler ------------------------------- #
    optimizer_cls = {"adam": optim.Adam, "radam": RAdam}.get(config.optimizer.lower())
    if optimizer_cls is None:
        raise NotImplementedError(f"Unknown optimizer: {config.optimizer}")
    optimizer = optimizer_cls(net.parameters(), lr=config.lr)

    scheduler = None
    if config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=config.lr_min)
    elif config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=config.lr_decay)
    elif config.lr_scheduler == "multi":
        milestones = [(x + 1) * 10000 // config.validation_interval for x in range(10)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=config.lr_decay)
    elif config.lr_scheduler != "no":
        raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")

    # ------------------------------- Trainer ------------------------------- #
    trainer = Trainer(
        net,
        criterion=loss_fn,
        metric=metric,
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        ckpt=config.ckpt,
        is_data_dict=True,
        experiment_id=os.path.splitext(os.path.basename(config.ckpt))[0],
        tensorboard_dir=config.tensorboard_dir,
        
    )

    # ------------------------------- Validation Callback ------------------------------- #
    # save_counter = 0
    # save_interval = 10

    log_data = []  # To store logs for CSV

    save_counter = 0
    save_interval = 10  # Already defined

    def validation_callback():
        global save_counter
        net.eval()
    
        # Evaluate metrics on validation data
        def evaluate_metrics(data_loader, phase):
            batch = next(iter(data_loader))
            batch = {k: v.to(device) for k, v in batch.items()}
            estimation = net(batch)
            loss = loss_fn(estimation, batch["audio"])
            metric_val = metric(estimation, batch["audio"])
    
            print(f"{phase.capitalize()} - Loss: {loss.item():.6f}, Metric: {metric_val.item():.6f}")
    
            return loss.item(), metric_val.item()
    
        # Log audio to tensorboard
        def log_audio(data_loader, phase):
            batch = next(iter(data_loader))
            batch = {k: v.to(device) for k, v in batch.items()}
            original_audio = batch["audio"][0]
            estimation = net(batch)
    
            if config.use_reverb:
                recon_reverb = estimation["audio_reverb"][0, :len(original_audio)]
                trainer.tensorboard.add_audio(
                    f"{trainer.config['experiment_id']}/{phase}_recon",
                    recon_reverb.cpu(),
                    trainer.config["step"],
                    sample_rate=config.sample_rate,
                )
    
            recon_synth = estimation["audio_synth"][0, :len(original_audio)]
            trainer.tensorboard.add_audio(
                f"{trainer.config['experiment_id']}/{phase}_recon_dereverb",
                recon_synth.cpu(),
                trainer.config["step"],
                sample_rate=config.sample_rate,
            )
    
            trainer.tensorboard.add_audio(
                f"{trainer.config['experiment_id']}/{phase}_original",
                original_audio.cpu(),
                trainer.config["step"],
                sample_rate=config.sample_rate,
            )
    
        # Get validation metrics
        val_loss, val_metric = evaluate_metrics(valid_loader, "valid")
    
        # You can optionally evaluate on a small batch of training data too:
        train_loss, train_metric = evaluate_metrics(train_loader, "train")
    
        # Save step metrics to log_data
        log_data.append({
            "step": trainer.config["step"],
            "train_loss": train_loss,
            "train_metric": train_metric,
            "val_loss": val_loss,
            "val_metric": val_metric,
        })
    
        # Periodically save CSV file
        if save_counter % save_interval == 0:
            df = pd.DataFrame(log_data)
            df.to_csv(f"{config.experiment_name}_metrics_log.csv", index=False)
            print(f"Saved metrics to {config.experiment_name}_metrics_log.csv")
    
        # Log audio samples
        log_audio(train_loader, "train")
        log_audio(valid_loader, "valid")
    
        # Periodically save model checkpoint
        save_counter += 1
        if save_counter % save_interval == 0:
            trainer.save(f"{trainer.ckpt}-{trainer.config['step']}")
    
    # Register the modified callback
    trainer.register_callback(validation_callback)

    if config.resume:
        trainer.load(config.ckpt)

    trainer.add_external_config(config)
    trainer.train(step=config.num_step, validation_interval=config.validation_interval)
