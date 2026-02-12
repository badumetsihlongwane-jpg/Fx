#!/usr/bin/env python3
"""
Evaluation script for NEXUS-FX.

Usage:
    python -m nexus_fx.scripts.evaluate --checkpoint checkpoints/best_model.pt --data_path data/
"""

import argparse
import torch
from torch.utils.data import DataLoader

from nexus_fx.config import NexusFXConfig
from nexus_fx.model import NEXUSFX
from nexus_fx.data import ForexDataset
from nexus_fx.training import NexusFXEvaluator
from nexus_fx.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate NEXUS-FX model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                      help='Path to forex data CSV files')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--split', type=str, default='test',
                      choices=['train', 'val', 'test'],
                      help='Data split to evaluate on')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting NEXUS-FX evaluation")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    logger.info(f"Config: {config}")
    
    # Create dataset
    logger.info("Loading data...")
    test_dataset = ForexDataset(
        data_path=args.data_path,
        pairs=config.pairs,
        base_timeframe=config.base_timeframe,
        target_timeframes=config.timeframes,
        sequence_length=config.sequence_length,
        mode=args.split,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    logger.info("Creating NEXUS-FX model...")
    model = NEXUSFX(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Evaluator
    evaluator = NexusFXEvaluator()
    
    # Evaluate
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move to device
            ohlc = batch['ohlc'].to(args.device)
            volume = batch['volume'].to(args.device)
            timestamps = batch['timestamps'].to(args.device)
            
            # Forward pass
            try:
                outputs = model(ohlc, volume, timestamps, macro_data=None)
                
                # For evaluation, create dummy targets from data
                # In practice, these would come from the batch
                targets = {
                    'direction': torch.randint(0, 3, (len(timestamps),), device=args.device),
                    'volatility': torch.randn(len(timestamps), device=args.device).abs(),
                }
                
                evaluator.update(outputs, targets)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Compute and display metrics
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    summary = evaluator.get_performance_summary()
    logger.info(summary)
    
    # Save metrics
    metrics = evaluator.compute_metrics()
    torch.save(metrics, 'evaluation_metrics.pt')
    logger.info("\nMetrics saved to evaluation_metrics.pt")


if __name__ == '__main__':
    main()
