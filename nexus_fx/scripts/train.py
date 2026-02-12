#!/usr/bin/env python3
"""
Main training script for NEXUS-FX.

Usage:
    python -m nexus_fx.scripts.train --config config.yaml
"""

import argparse
import torch
from torch.utils.data import DataLoader

from nexus_fx.config import NexusFXConfig
from nexus_fx.model import NEXUSFX
from nexus_fx.data import ForexDataset, Preprocessor
from nexus_fx.training import NexusFXTrainer
from nexus_fx.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train NEXUS-FX model')
    parser.add_argument('--data_path', type=str, default=None,
                      help='Path to forex data CSV files')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting NEXUS-FX training")
    
    # Create config
    config = NexusFXConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        device=args.device,
    )
    
    logger.info(f"Config: {config}")
    
    # Create datasets
    logger.info("Loading data...")
    train_dataset = ForexDataset(
        data_path=args.data_path,
        pairs=config.pairs,
        base_timeframe=config.base_timeframe,
        target_timeframes=config.timeframes,
        sequence_length=config.sequence_length,
        mode='train',
    )
    
    val_dataset = ForexDataset(
        data_path=args.data_path,
        pairs=config.pairs,
        base_timeframe=config.base_timeframe,
        target_timeframes=config.timeframes,
        sequence_length=config.sequence_length,
        mode='val',
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating NEXUS-FX model...")
    model = NEXUSFX(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = NexusFXTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
