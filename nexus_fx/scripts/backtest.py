#!/usr/bin/env python3
"""
Simple backtesting framework for NEXUS-FX.

Usage:
    python -m nexus_fx.scripts.backtest --checkpoint checkpoints/best_model.pt --data_path data/
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from nexus_fx.config import NexusFXConfig
from nexus_fx.model import NEXUSFX
from nexus_fx.data import ForexDataset
from nexus_fx.utils import setup_logger, calculate_spread


class SimpleBacktester:
    """
    Simple event-driven backtester for NEXUS-FX.
    
    Features:
    - Position sizing based on model confidence
    - Spread/commission modeling
    - Session-aware execution
    - Performance tracking
    """
    
    def __init__(
        self,
        model,
        config: NexusFXConfig,
        initial_capital: float = 10000.0,
        max_position_size: float = 0.02,  # 2% per trade
        device: str = 'cuda',
    ):
        self.model = model
        self.config = config
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.device = device
        
        # State
        self.capital = initial_capital
        self.positions = {pair: 0.0 for pair in config.pairs}
        self.trades = []
        self.equity_curve = [initial_capital]
    
    def run(self, dataloader: DataLoader) -> dict:
        """Run backtest on dataloader"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                ohlc = batch['ohlc'].to(self.device)
                volume = batch['volume'].to(self.device)
                timestamps = batch['timestamps'].to(self.device)
                
                # Get predictions
                outputs = self.model(ohlc, volume, timestamps, macro_data=None)
                predictions = self.model.output_heads.get_predictions(outputs)
                
                # Execute trades based on predictions
                self._execute_predictions(predictions, ohlc, timestamps)
        
        # Compute performance metrics
        return self._compute_performance()
    
    def _execute_predictions(self, predictions, ohlc, timestamps):
        """Execute trades based on model predictions"""
        # Get direction and confidence
        direction = predictions['direction_class'].cpu().numpy()
        confidence = predictions['confidence'].cpu().numpy()
        
        # For simplicity, trade on the first pair only
        pair_idx = 0
        pair = self.config.pairs[pair_idx]
        
        # Current price (close of last candle)
        current_price = ohlc[0, pair_idx, 0, -1, 3].item()  # Close price
        
        # Calculate position size based on confidence
        position_size = self.capital * self.max_position_size * float(confidence[0])
        
        # Calculate spread
        spread = calculate_spread(pair) * 0.0001  # Convert pips to price
        
        # Trade decision
        if direction[0] == 2 and self.positions[pair] <= 0:  # Buy signal
            # Close short if any
            if self.positions[pair] < 0:
                pnl = -self.positions[pair] * (current_price - self.positions[pair + '_entry'])
                self.capital += pnl
                self.positions[pair] = 0
            
            # Open long
            shares = position_size / (current_price + spread)
            self.positions[pair] = shares
            self.positions[pair + '_entry'] = current_price + spread
            
            self.trades.append({
                'pair': pair,
                'type': 'long',
                'price': current_price + spread,
                'size': shares,
                'capital': self.capital,
            })
        
        elif direction[0] == 0 and self.positions[pair] >= 0:  # Sell signal
            # Close long if any
            if self.positions[pair] > 0:
                pnl = self.positions[pair] * (current_price - spread - self.positions[pair + '_entry'])
                self.capital += pnl
                self.positions[pair] = 0
            
            # Open short
            shares = position_size / (current_price - spread)
            self.positions[pair] = -shares
            self.positions[pair + '_entry'] = current_price - spread
            
            self.trades.append({
                'pair': pair,
                'type': 'short',
                'price': current_price - spread,
                'size': shares,
                'capital': self.capital,
            })
        
        # Update equity curve
        self.equity_curve.append(self.capital)
    
    def _compute_performance(self) -> dict:
        """Compute backtest performance metrics"""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        metrics = {
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'num_trades': len(self.trades),
            'sharpe_ratio': returns.mean() / (returns.std() + 1e-8) * np.sqrt(252),
            'max_drawdown': ((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)).min(),
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Backtest NEXUS-FX model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default=None,
                      help='Path to forex data CSV files')
    parser.add_argument('--initial_capital', type=float, default=10000.0,
                      help='Initial capital for backtesting')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting NEXUS-FX backtesting")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    # Create test dataset
    logger.info("Loading test data...")
    test_dataset = ForexDataset(
        data_path=args.data_path,
        pairs=config.pairs,
        base_timeframe=config.base_timeframe,
        target_timeframes=config.timeframes,
        sequence_length=config.sequence_length,
        mode='test',
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for backtesting
        shuffle=False,
    )
    
    # Create model
    logger.info("Creating NEXUS-FX model...")
    model = NEXUSFX(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    # Create backtester
    backtester = SimpleBacktester(
        model=model,
        config=config,
        initial_capital=args.initial_capital,
        device=args.device,
    )
    
    # Run backtest
    logger.info("Running backtest...")
    metrics = backtester.run(test_loader)
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Initial Capital: ${args.initial_capital:,.2f}")
    logger.info(f"Final Capital: ${metrics['final_capital']:,.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Number of Trades: {metrics['num_trades']}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
