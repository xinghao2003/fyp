import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ModelComparisonVisualizer:
    """
    Enhanced visualization suite for comparing DRL model with baseline strategies
    """

    def __init__(self, figsize=(12, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def load_and_prepare_data(self, folder_path):
        """Load and combine all strategy results"""
        folder = Path(folder_path)
        csv_files = list(folder.glob('*.csv'))

        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                strategy = self._extract_strategy_name(csv_file.name)
                df['Strategy'] = strategy
                df['Symbol'] = df['File'].apply(self._extract_symbol)

                # Convert numeric columns to proper data types
                numeric_columns = [
                    'Return [%]', 'Sharpe Ratio', 'Volatility (Ann.) [%]',
                    'Max. Drawdown [%]', 'Win Rate [%]', 'Profit Factor',
                    'Calmar Ratio', 'Sortino Ratio', 'CAGR [%]', 'Return (Ann.) [%]',
                    'Alpha [%]', 'Beta', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', '# Trades', 'Best Trade [%]',
                    'Worst Trade [%]', 'Avg. Trade [%]', 'Expectancy [%]', 'SQN'
                ]

                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                all_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _extract_strategy_name(self, filename):
        """Extract strategy name from CSV filename"""
        if filename.startswith('rl_backtest_summary_'):
            return 'DRL Model'
        elif 'BuyAndHold' in filename:
            return 'Buy & Hold'
        elif 'SmaX' in filename:
            return 'SMA Crossover'
        elif 'RsiReversion' in filename:
            return 'RSI Mean Reversion'
        elif 'BollingerBands' in filename:
            return 'Bollinger Bands'
        elif 'DonchianChannel' in filename:
            return 'Donchian Breakout'
        else:
            return filename.split('_')[2] if '_' in filename else filename.replace('.csv', '')

    def _extract_symbol(self, filename):
        """Extract symbol from filename"""
        base_name = filename.replace('.csv', '')
        if '_USD-1d-max' in base_name:
            return base_name.split('_USD-1d-max')[0]
        elif '-USD_USD-1d-max' in base_name:
            return base_name.split('-USD_USD-1d-max')[0]
        else:
            return base_name.split('_')[0]

    def create_performance_radar_chart(self, df, metrics=None, save_path=None):
        """
        Create radar chart comparing strategies across multiple metrics
        """
        if metrics is None:
            metrics = ['Sharpe Ratio', 'Return [%]', 'Win Rate [%]',
                       'Profit Factor', 'Calmar Ratio']

        # Filter out rows with missing data for the selected metrics
        df_clean = df.dropna(subset=metrics)

        if df_clean.empty:
            print("No data available for radar chart after cleaning")
            return None

        # Aggregate by strategy (mean across all symbols)
        strategy_means = df_clean.groupby('Strategy')[metrics].mean()

        # Normalize metrics to 0-1 scale for radar chart
        normalized_data = strategy_means.copy()
        for metric in metrics:
            min_val = strategy_means[metric].min()
            max_val = strategy_means[metric].max()
            if max_val != min_val:
                normalized_data[metric] = (
                    strategy_means[metric] - min_val) / (max_val - min_val)
            else:
                normalized_data[metric] = 0.5

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics),
                             endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=self.figsize,
                               subplot_kw=dict(projection='polar'))

        for i, (strategy, values) in enumerate(normalized_data.iterrows()):
            values_list = values.tolist()
            values_list += values_list[:1]  # Complete the circle

            ax.plot(angles, values_list, 'o-', linewidth=2,
                    label=strategy, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values_list, alpha=0.25,
                    color=self.colors[i % len(self.colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(
            'Strategy Performance Comparison (Normalized)', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig

    def create_risk_return_scatter(self, df, save_path=None):
        """
        Create risk-return scatter plot with Sharpe ratio as color intensity
        """
        required_cols = [
            'Return [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Max. Drawdown [%]']
        df_clean = df.dropna(subset=required_cols)

        if df_clean.empty:
            print("No data available for risk-return scatter after cleaning")
            return None

        strategy_stats = df_clean.groupby('Strategy').agg({
            'Return [%]': 'mean',
            'Volatility (Ann.) [%]': 'mean',
            'Sharpe Ratio': 'mean',
            'Max. Drawdown [%]': 'mean'
        }).reset_index()

        fig, ax = plt.subplots(figsize=self.figsize)

        scatter = ax.scatter(strategy_stats['Volatility (Ann.) [%]'],
                             strategy_stats['Return [%]'],
                             c=strategy_stats['Sharpe Ratio'],
                             s=200, alpha=0.7, cmap='viridis',
                             edgecolors='black', linewidth=1)

        # Add strategy labels
        for i, row in strategy_stats.iterrows():
            ax.annotate(row['Strategy'],
                        (row['Volatility (Ann.) [%]'], row['Return [%]']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')

        ax.set_xlabel('Volatility (Annualized) [%]', fontsize=12)
        ax.set_ylabel('Return [%]', fontsize=12)
        ax.set_title('Risk-Return Profile by Strategy\n(Color intensity = Sharpe Ratio)',
                     fontsize=14, pad=20)

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio', fontsize=12)

        # Add diagonal lines for reference Sharpe ratios
        x_range = ax.get_xlim()
        for sharpe in [0.5, 1.0, 1.5]:
            x_vals = np.linspace(x_range[0], x_range[1], 100)
            y_vals = sharpe * x_vals
            ax.plot(x_vals, y_vals, '--', alpha=0.3, color='gray')
            ax.text(x_range[1] * 0.8, sharpe * x_range[1] * 0.8,
                    f'Sharpe = {sharpe}', alpha=0.5, fontsize=8)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig

    def create_performance_heatmap(self, df, metric='Sharpe Ratio', save_path=None):
        """
        Create heatmap showing strategy performance across different assets
        """
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in data")
            return None

        df_clean = df.dropna(subset=[metric])

        if df_clean.empty:
            print(f"No data available for {metric} heatmap after cleaning")
            return None

        pivot_df = df_clean.pivot(
            index='Symbol', columns='Strategy', values=metric)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, square=False, ax=ax, cbar_kws={'label': metric})

        ax.set_title(f'{metric} Heatmap: Strategy vs Asset Performance',
                     fontsize=14, pad=20)
        ax.set_xlabel('Strategy', fontsize=12)
        ax.set_ylabel('Asset Symbol', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig

    def create_metric_comparison_bars(self, df, metrics=None, save_path=None):
        """
        Create grouped bar chart for multiple metrics comparison
        """
        if metrics is None:
            metrics = ['Return [%]', 'Sharpe Ratio',
                       'Win Rate [%]', 'Max. Drawdown [%]']

        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            print("None of the specified metrics are available in the data")
            return None

        df_clean = df.dropna(subset=available_metrics)

        if df_clean.empty:
            print("No data available for metric comparison after cleaning")
            return None

        strategy_means = df_clean.groupby('Strategy')[available_metrics].mean()

        # Create subplots based on available metrics
        n_metrics = len(available_metrics)
        rows = (n_metrics + 1) // 2
        cols = 2 if n_metrics > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.ravel()

        for i, metric in enumerate(available_metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break

            # Sort strategies by performance for this metric
            sorted_data = strategy_means[metric].sort_values(ascending=False)

            bars = ax.bar(range(len(sorted_data)), sorted_data.values,
                          color=[self.colors[j % len(self.colors)] for j in range(len(sorted_data))])

            ax.set_title(f'{metric} by Strategy',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticks(range(len(sorted_data)))
            ax.set_xticklabels(sorted_data.index, rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, sorted_data.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)

            ax.grid(True, alpha=0.3)

        # Hide extra subplots if any
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Strategy Performance Comparison Across Key Metrics',
                     fontsize=16, y=0.98)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig

    def create_drawdown_comparison(self, df, save_path=None):
        """
        Create drawdown comparison visualization
        """
        required_cols = ['Max. Drawdown [%]', 'Max. Drawdown Duration']
        df_clean = df.dropna(subset=required_cols)

        if df_clean.empty:
            print("No data available for drawdown comparison after cleaning")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Max Drawdown comparison
        drawdown_data = df_clean.groupby(
            'Strategy')['Max. Drawdown [%]'].mean().sort_values()
        bars1 = ax1.barh(range(len(drawdown_data)), drawdown_data.values,
                         color=[self.colors[i % len(self.colors)] for i in range(len(drawdown_data))])
        ax1.set_title('Maximum Drawdown by Strategy',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Max Drawdown [%]', fontsize=10)
        ax1.set_yticks(range(len(drawdown_data)))
        ax1.set_yticklabels(drawdown_data.index)

        # Add value labels
        for bar, value in zip(bars1, drawdown_data.values):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f'{value:.1f}%', ha='left', va='center', fontsize=9)

        # Drawdown Duration comparison
        duration_data = df_clean.groupby('Strategy')[
            'Max. Drawdown Duration'].mean().sort_values()
        bars2 = ax2.barh(range(len(duration_data)), duration_data.values,
                         color=[self.colors[i % len(self.colors)] for i in range(len(duration_data))])
        ax2.set_title('Max Drawdown Duration by Strategy',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('Duration (Days)', fontsize=10)
        ax2.set_yticks(range(len(duration_data)))
        ax2.set_yticklabels(duration_data.index)

        # Add value labels
        for bar, value in zip(bars2, duration_data.values):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{value:.0f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig

    def create_statistical_significance_test(self, df, save_path=None):
        """
        Create statistical significance comparison between DRL model and baselines
        """
        from scipy import stats

        if 'Return [%]' not in df.columns:
            print("Return [%] column not found for statistical testing")
            return None, None

        df_clean = df.dropna(subset=['Return [%]'])

        if df_clean.empty:
            print("No return data available for statistical testing")
            return None, None

        drl_returns = df_clean[df_clean['Strategy']
                               == 'DRL Model']['Return [%]'].dropna()

        if len(drl_returns) == 0:
            print("No DRL Model returns found for comparison")
            return None, None

        baseline_comparisons = []

        strategies = df_clean['Strategy'].unique()
        for strategy in strategies:
            if strategy != 'DRL Model':
                baseline_returns = df_clean[df_clean['Strategy']
                                            == strategy]['Return [%]'].dropna()
                if len(baseline_returns) > 1 and len(drl_returns) > 1:
                    t_stat, p_value = stats.ttest_ind(
                        drl_returns, baseline_returns)
                    baseline_comparisons.append({
                        'Strategy': strategy,
                        'T-Statistic': t_stat,
                        'P-Value': p_value,
                        'Significant': p_value < 0.05,
                        'DRL_Mean': drl_returns.mean(),
                        'Baseline_Mean': baseline_returns.mean()
                    })

        comparison_df = pd.DataFrame(baseline_comparisons)

        if len(comparison_df) == 0:
            print("No statistical comparisons could be performed")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # T-statistics plot
        colors = ['red' if sig else 'blue' for sig in comparison_df['Significant']]
        bars1 = ax1.bar(range(len(comparison_df)), comparison_df['T-Statistic'],
                        color=colors, alpha=0.7)
        ax1.set_title('T-Statistics: DRL Model vs Baselines',
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel('T-Statistic', fontsize=10)
        ax1.set_xticks(range(len(comparison_df)))
        ax1.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # P-values plot
        bars2 = ax2.bar(range(len(comparison_df)), comparison_df['P-Value'],
                        color=colors, alpha=0.7)
        ax2.set_title('P-Values: DRL Model vs Baselines',
                      fontsize=12, fontweight='bold')
        ax2.set_ylabel('P-Value', fontsize=10)
        ax2.set_xticks(range(len(comparison_df)))
        ax2.set_xticklabels(comparison_df['Strategy'], rotation=45, ha='right')
        ax2.axhline(y=0.05, color='red', linestyle='--',
                    alpha=0.7, label='α = 0.05')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig, comparison_df

    def create_comprehensive_summary_table(self, df, save_path=None):
        """
        Create a comprehensive summary table for the report
        """
        key_metrics = ['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]',
                       'Win Rate [%]', 'Profit Factor', 'Calmar Ratio']

        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in key_metrics if m in df.columns]

        if not available_metrics:
            print("None of the key metrics are available in the data")
            return None

        df_clean = df.dropna(subset=available_metrics)

        if df_clean.empty:
            print("No data available for summary table after cleaning")
            return None

        summary = df_clean.groupby('Strategy')[available_metrics].agg([
            'mean', 'std']).round(3)

        # Flatten column names
        summary.columns = [
            f'{metric}_{stat}' for metric, stat in summary.columns]

        # Add ranking for each metric
        for metric in key_metrics:
            if 'Drawdown' in metric:  # Lower is better for drawdown
                summary[f'{metric}_rank'] = summary[f'{metric}_mean'].rank()
            else:  # Higher is better for other metrics
                summary[f'{metric}_rank'] = summary[f'{metric}_mean'].rank(
                    ascending=False)

        # Calculate overall score (average rank)
        rank_columns = [col for col in summary.columns if '_rank' in col]
        summary['Overall_Rank'] = summary[rank_columns].mean(axis=1)
        summary['Overall_Score'] = summary['Overall_Rank'].rank()

        print("COMPREHENSIVE STRATEGY COMPARISON SUMMARY")
        print("="*60)
        print(summary.to_string())

        if save_path:
            summary.to_csv(save_path.replace('.png', '.csv'))

        return summary

# Example usage and recommended metrics


def get_recommended_metrics():
    """
    Return recommended metrics for trading strategy evaluation
    """
    return {
        'risk_adjusted_returns': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
        'return_metrics': ['Return [%]', 'CAGR [%]', 'Return (Ann.) [%]'],
        'risk_metrics': ['Max. Drawdown [%]', 'Volatility (Ann.) [%]', 'Avg. Drawdown [%]'],
        'trade_efficiency': ['Win Rate [%]', 'Profit Factor', 'Avg. Trade [%]'],
        'robustness': ['# Trades', 'SQN', 'Expectancy [%]']
    }


# Main execution example
if __name__ == "__main__":
    # Initialize visualizer
    viz = ModelComparisonVisualizer(figsize=(12, 8), dpi=300)

    # Load your data (replace with your actual folder path)
    folder_path = r"C:\Users\xingh\Desktop\fyp-code\backtesting\result\plot"

    try:
        # Load data
        df = viz.load_and_prepare_data(folder_path)

        if df.empty:
            print("No data loaded. Please check your folder path and CSV files.")
        else:
            print(f"Loaded data for {len(df)} strategy-asset combinations")
            print(f"Strategies: {df['Strategy'].unique()}")
            print(f"Assets: {df['Symbol'].unique()}")

            # Check available numeric columns
            numeric_cols = df.select_dtypes(
                include=[np.number]).columns.tolist()
            print(f"Available numeric columns: {numeric_cols}")

            # Create all visualizations with error handling
            print("\nCreating visualizations...")

            # 1. Performance radar chart
            try:
                viz.create_performance_radar_chart(
                    df, save_path="performance_radar.png")
            except Exception as e:
                print(f"Error creating radar chart: {e}")

            # 2. Risk-return scatter
            try:
                viz.create_risk_return_scatter(
                    df, save_path="risk_return_scatter.png")
            except Exception as e:
                print(f"Error creating risk-return scatter: {e}")

            # 3. Performance heatmap
            try:
                viz.create_performance_heatmap(df, metric='Sharpe Ratio',
                                               save_path="sharpe_heatmap.png")
            except Exception as e:
                print(f"Error creating heatmap: {e}")

            # 4. Metric comparison bars
            try:
                viz.create_metric_comparison_bars(
                    df, save_path="metric_comparison.png")
            except Exception as e:
                print(f"Error creating metric comparison: {e}")

            # 5. Drawdown comparison
            try:
                viz.create_drawdown_comparison(
                    df, save_path="drawdown_comparison.png")
            except Exception as e:
                print(f"Error creating drawdown comparison: {e}")

            # 6. Statistical significance test
            try:
                fig, stats_df = viz.create_statistical_significance_test(
                    df, save_path="statistical_significance.png")
            except Exception as e:
                print(f"Error creating statistical significance test: {e}")

            # 7. Comprehensive summary table
            try:
                summary = viz.create_comprehensive_summary_table(
                    df, save_path="comprehensive_summary.png")
            except Exception as e:
                print(f"Error creating summary table: {e}")

            print("\nVisualization creation completed!")

            # Recommended metrics for the report
            metrics = get_recommended_metrics()
            print("Recommended metrics for your report:")
            for category, metric_list in metrics.items():
                print(f"\n{category.replace('_', ' ').title()}:")
                for metric in metric_list:
                    print(f"  - {metric}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure your CSV files are in the correct format and folder path is valid.")
