"""
Forex-specific evaluation metrics.

Metrics include:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Win rate
- Profit factor
- Per-regime performance
- Calibration metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class NexusFXEvaluator:
    """
    Evaluator for forex trading performance.
    
    Computes both prediction accuracy metrics and trading performance metrics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated statistics"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.returns = []
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        returns: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update with new batch of predictions.
        
        Args:
            predictions: Model predictions dict
            targets: Ground truth targets dict
            returns: Actual returns (optional, for trading metrics)
        """
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if 'confidence' in predictions:
            self.confidences.append(predictions['confidence'].cpu())
        
        if returns is not None:
            self.returns.append(returns.cpu())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            metrics: Dictionary of metric name -> value
        """
        metrics = {}
        
        # Prediction accuracy metrics
        acc_metrics = self._compute_accuracy_metrics()
        metrics.update(acc_metrics)
        
        # Trading performance metrics
        if len(self.returns) > 0:
            trading_metrics = self._compute_trading_metrics()
            metrics.update(trading_metrics)
        
        # Calibration metrics
        if len(self.confidences) > 0:
            cal_metrics = self._compute_calibration_metrics()
            metrics.update(cal_metrics)
        
        return metrics
    
    def _compute_accuracy_metrics(self) -> Dict[str, float]:
        """Compute prediction accuracy metrics"""
        metrics = {}
        
        # Direction accuracy
        all_pred_classes = []
        all_target_classes = []
        
        for pred, target in zip(self.predictions, self.targets):
            if 'direction_logits' in pred and 'direction' in target:
                pred_class = torch.argmax(pred['direction_logits'], dim=-1)
                all_pred_classes.append(pred_class.cpu())
                all_target_classes.append(target['direction'].cpu())
        
        if all_pred_classes:
            pred_classes = torch.cat(all_pred_classes)
            target_classes = torch.cat(all_target_classes)
            
            accuracy = (pred_classes == target_classes).float().mean().item()
            metrics['direction_accuracy'] = accuracy
            
            # Per-class accuracy
            for i in range(3):  # Assuming 3 classes
                mask = target_classes == i
                if mask.sum() > 0:
                    class_acc = (pred_classes[mask] == target_classes[mask]).float().mean().item()
                    metrics[f'direction_accuracy_class_{i}'] = class_acc
        
        return metrics
    
    def _compute_trading_metrics(self) -> Dict[str, float]:
        """Compute trading performance metrics"""
        metrics = {}
        
        # Concatenate all returns
        all_returns = torch.cat(self.returns).numpy()
        
        # Cumulative returns
        cumulative_returns = np.cumprod(1 + all_returns) - 1
        total_return = cumulative_returns[-1]
        metrics['total_return'] = total_return
        
        # Sharpe ratio (annualized, assuming 5-min returns)
        # 252 trading days * 24 hours * 12 (5-min periods per hour)
        periods_per_year = 252 * 24 * 12
        sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(periods_per_year)
        metrics['sharpe_ratio'] = sharpe
        
        # Sortino ratio (only downside volatility)
        downside_returns = all_returns[all_returns < 0]
        if len(downside_returns) > 0:
            sortino = np.mean(all_returns) / (np.std(downside_returns) + 1e-8) * np.sqrt(periods_per_year)
            metrics['sortino_ratio'] = sortino
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + all_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        metrics['max_drawdown'] = max_drawdown
        
        # Win rate
        winning_trades = (all_returns > 0).sum()
        total_trades = len(all_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        metrics['win_rate'] = win_rate
        
        # Profit factor
        gross_profit = all_returns[all_returns > 0].sum()
        gross_loss = -all_returns[all_returns < 0].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        metrics['profit_factor'] = profit_factor
        
        return metrics
    
    def _compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute calibration metrics"""
        metrics = {}
        
        # Confidence calibration
        all_confidences = torch.cat(self.confidences)
        all_pred_classes = []
        all_target_classes = []
        
        for pred, target in zip(self.predictions, self.targets):
            if 'direction_logits' in pred and 'direction' in target:
                pred_class = torch.argmax(pred['direction_logits'], dim=-1)
                all_pred_classes.append(pred_class.cpu())
                all_target_classes.append(target['direction'].cpu())
        
        if all_pred_classes:
            pred_classes = torch.cat(all_pred_classes)
            target_classes = torch.cat(all_target_classes)
            correct = (pred_classes == target_classes).float()
            
            # Expected Calibration Error (ECE)
            num_bins = 10
            bin_boundaries = torch.linspace(0, 1, num_bins + 1)
            ece = 0.0
            
            for i in range(num_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (all_confidences >= bin_lower) & (all_confidences < bin_upper)
                in_bin = in_bin.squeeze()
                
                if in_bin.sum() > 0:
                    bin_confidence = all_confidences[in_bin].mean()
                    bin_accuracy = correct[in_bin].mean()
                    ece += torch.abs(bin_confidence - bin_accuracy) * (in_bin.sum() / len(all_confidences))
            
            metrics['expected_calibration_error'] = ece.item()
        
        return metrics
    
    def get_performance_summary(self) -> str:
        """Get formatted performance summary"""
        metrics = self.compute_metrics()
        
        summary = "=== NEXUS-FX Performance Summary ===\n\n"
        
        summary += "Prediction Metrics:\n"
        summary += f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.4f}\n"
        
        summary += "\nTrading Metrics:\n"
        summary += f"  Total Return: {metrics.get('total_return', 0):.4f}\n"
        summary += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n"
        summary += f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}\n"
        summary += f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f}\n"
        summary += f"  Win Rate: {metrics.get('win_rate', 0):.4f}\n"
        summary += f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}\n"
        
        summary += "\nCalibration Metrics:\n"
        summary += f"  Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}\n"
        
        return summary
