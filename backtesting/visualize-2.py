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

                # Handle Buy & Hold specific metrics
                df = self._handle_buy_hold_metrics(df, strategy)

                all_data.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _handle_buy_hold_metrics(self, df, strategy):
        """Handle missing metrics for Buy & Hold strategy"""
        if strategy == 'Buy & Hold':
            # Fill trading-specific metrics that don't apply to Buy & Hold
            trading_metrics = {
                # Buy & Hold is always "winning" if return > 0
                'Win Rate [%]': 100.0,
                '# Trades': 1,  # Conceptually one trade (buy and hold)
                'Profit Factor': np.nan,  # Not meaningful for Buy & Hold
                'SQN': np.nan,  # System Quality Number not applicable
                'Avg. Trade [%]': df['Return [%]'].iloc[0] if not df.empty else np.nan,
                'Best Trade [%]': df['Return [%]'].iloc[0] if not df.empty else np.nan,
                'Worst Trade [%]': df['Return [%]'].iloc[0] if not df.empty else np.nan,
                'Expectancy [%]': df['Return [%]'].iloc[0] if not df.empty else np.nan
            }

            for metric, value in trading_metrics.items():
                if metric in df.columns and df[metric].isna().all():
                    df[metric] = value

        return df

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

    def _get_cleaned_data(self, df, required_metrics, strategy_specific=True):
        """Get cleaned data with strategy-specific handling"""
        if not strategy_specific:
            # For visualizations that need all strategies, use more flexible filtering
            df_clean = df.copy()

            # Only require core metrics that all strategies should have
            core_metrics = ['Return [%]', 'Strategy', 'Symbol']
            available_core = [m for m in core_metrics if m in df.columns]
            df_clean = df_clean.dropna(subset=available_core)

            # For other metrics, fill NaN with strategy-appropriate values
            for metric in required_metrics:
                if metric in df_clean.columns:
                    # Fill NaN values based on strategy type
                    for strategy in df_clean['Strategy'].unique():
                        mask = df_clean['Strategy'] == strategy
                        if strategy == 'Buy & Hold' and df_clean.loc[mask, metric].isna().any():
                            if 'Win Rate' in metric:
                                df_clean.loc[mask, metric] = df_clean.loc[mask, metric].fillna(
                                    100.0)
                            elif '# Trades' in metric:
                                df_clean.loc[mask, metric] = df_clean.loc[mask, metric].fillna(
                                    1)
                            elif 'Profit Factor' in metric or 'SQN' in metric:
                                # Keep as NaN for Buy & Hold for these metrics
                                continue
                            else:
                                # For other metrics, use median of non-Buy & Hold strategies
                                other_strategies_median = df_clean[df_clean['Strategy'] != 'Buy & Hold'][metric].median(
                                )
                                if pd.notna(other_strategies_median):
                                    df_clean.loc[mask, metric] = df_clean.loc[mask, metric].fillna(
                                        other_strategies_median)

            return df_clean
        else:
            # Original behavior for strategy-specific analysis
            return df.dropna(subset=required_metrics)

    def create_performance_radar_chart(self, df, metrics=None, save_path=None):
        """
        Create radar chart comparing strategies across multiple metrics
        """
        if metrics is None:
            # Use metrics that are meaningful for all strategies including Buy & Hold
            metrics = ['Sharpe Ratio', 'Return [%]', 'Calmar Ratio',
                       'Max. Drawdown [%]', 'Volatility (Ann.) [%]']

        # Use flexible data cleaning
        df_clean = self._get_cleaned_data(df, metrics, strategy_specific=False)

        if df_clean.empty:
            print("No data available for radar chart after cleaning")
            return None

        # Aggregate by strategy (mean across all symbols)
        strategy_means = df_clean.groupby('Strategy')[metrics].mean()

        # Remove strategies with too many NaN values
        strategy_means = strategy_means.dropna(thresh=len(
            metrics)*0.6)  # Require at least 60% of metrics

        if strategy_means.empty:
            print("No strategies have sufficient data for radar chart")
            return None

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

        # Use more flexible filtering for heatmap
        df_clean = df.copy()

        # Only remove rows where the specific metric AND core identifiers are missing
        df_clean = df_clean.dropna(subset=[metric, 'Strategy', 'Symbol'])

        if df_clean.empty:
            print(f"No data available for {metric} heatmap after cleaning")
            return None

        pivot_df = df_clean.pivot(
            index='Symbol', columns='Strategy', values=metric)

        # Fill any remaining NaN values for better visualization
        # You might want to comment this out if you prefer to show NaN as blank
        # pivot_df = pivot_df.fillna(0)

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
            # Include both core and trading-specific metrics
            metrics = ['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', 'Volatility (Ann.) [%]',
                       'Win Rate [%]', 'Profit Factor', '# Trades', 'Expectancy [%]']

        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics if m in df.columns]

        if not available_metrics:
            print("None of the specified metrics are available in the data")
            return None

        # Use flexible data cleaning
        df_clean = self._get_cleaned_data(
            df, available_metrics, strategy_specific=False)

        if df_clean.empty:
            print("No data available for metric comparison after cleaning")
            return None

        # Create individual plots for each metric (with strategy filtering per metric)
        individual_figs = []
        for metric in available_metrics:
            individual_fig = self._create_individual_metric_plot(
                df_clean, metric, save_path)
            if individual_fig:
                individual_figs.append(individual_fig)

        # For the combined subplot, prepare data for each metric separately
        strategy_means = df_clean.groupby('Strategy')[available_metrics].mean()

        # Only include strategies that have data for at least half of the metrics
        strategy_means = strategy_means.dropna(
            thresh=len(available_metrics)*0.3)  # Reduced threshold to accommodate Buy & Hold

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

            # Filter strategies for trading-specific metrics
            metric_data = self._get_metric_data_for_subplot(df_clean, metric)

            if metric_data.empty:
                ax.text(0.5, 0.5, f'No data available\nfor {metric}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title(f'{metric} by Strategy',
                             fontsize=12, fontweight='bold')
                continue

            # Sort strategies by performance for this metric
            sorted_data = metric_data.sort_values(ascending=False)

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

        return fig, individual_figs

    def _get_metric_data_for_subplot(self, df_clean, metric):
        """Get cleaned metric data, filtering out strategies with no meaningful data for specific metrics"""
        # Trading-specific metrics where Buy & Hold should be excluded
        trading_specific_metrics = [
            'Win Rate [%]', 'Profit Factor', '# Trades', 'Expectancy [%]', 'SQN']

        metric_data = df_clean.groupby('Strategy')[metric].mean()

        if metric in trading_specific_metrics:
            # For trading-specific metrics, exclude Buy & Hold and strategies with NaN values
            metric_data = metric_data.dropna()
            if 'Buy & Hold' in metric_data.index:
                metric_data = metric_data.drop('Buy & Hold')
        else:
            # For other metrics, just drop NaN values
            metric_data = metric_data.dropna()

        return metric_data

    def _create_individual_metric_plot(self, df_clean, metric, base_save_path=None):
        """
        Create individual plot for a single metric
        """
        try:
            # Get metric data with appropriate strategy filtering
            metric_data = self._get_metric_data_for_subplot(df_clean, metric)

            if metric_data.empty:
                print(f"No data available for individual plot of {metric}")
                return None

            # Sort strategies by performance for this metric
            sorted_data = metric_data.sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(range(len(sorted_data)), sorted_data.values,
                          color=[self.colors[j % len(self.colors)] for j in range(len(sorted_data))])

            ax.set_title(f'{metric} Comparison Across Strategies',
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xlabel('Strategy', fontsize=12)
            ax.set_xticks(range(len(sorted_data)))
            ax.set_xticklabels(sorted_data.index, rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, sorted_data.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)

            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')

            # Highlight best performing strategy
            best_idx = 0  # Already sorted in descending order
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)

            # Add note for trading-specific metrics
            trading_specific_metrics = [
                'Win Rate [%]', 'Profit Factor', '# Trades', 'Expectancy [%]', 'SQN']
            if metric in trading_specific_metrics:
                # Place note at top right and explain why Buy & Hold is excluded
                ax.text(0.98, 0.98,
                        'Note: Buy & Hold excluded for this metric\n(Buy & Hold does not generate trades, so trade-based metrics are not meaningful)',
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

            plt.tight_layout()

            # Save individual plot
            if base_save_path:
                # Create individual filename
                metric_clean = metric.replace('[%]', '').replace(
                    '(', '').replace(')', '').replace(' ', '_').replace('.', '').replace('#', 'num')
                individual_save_path = base_save_path.replace(
                    '.png', f'_{metric_clean.lower()}.png')
                plt.savefig(individual_save_path,
                            dpi=self.dpi, bbox_inches='tight')
                print(f"Saved individual plot: {individual_save_path}")

            plt.show()
            return fig

        except Exception as e:
            print(f"Error creating individual plot for {metric}: {e}")
            return None

    def create_drawdown_comparison(self, df, save_path=None):
        """
        Create drawdown comparison visualization
        """
        required_cols = ['Max. Drawdown [%]', 'Max. Drawdown Duration']

        # Use more flexible data cleaning for drawdown comparison
        df_clean = self._get_cleaned_data(
            df, required_cols, strategy_specific=False)

        if df_clean.empty:
            print("No data available for drawdown comparison after cleaning")
            return None

        # Check if we have the required columns after cleaning
        available_cols = [
            col for col in required_cols if col in df_clean.columns and not df_clean[col].isna().all()]

        if not available_cols:
            print("No drawdown data available after cleaning")
            return None

        fig_width = 16 if len(available_cols) == 2 else 8
        fig, axes = plt.subplots(
            1, len(available_cols), figsize=(fig_width, 6))

        if len(available_cols) == 1:
            axes = [axes]

        plot_idx = 0

        # Max Drawdown comparison
        if 'Max. Drawdown [%]' in available_cols:
            ax = axes[plot_idx]
            drawdown_data = df_clean.groupby(
                'Strategy')['Max. Drawdown [%]'].mean().dropna().sort_values()

            if not drawdown_data.empty:
                bars = ax.barh(range(len(drawdown_data)), drawdown_data.values,
                               color=[self.colors[i % len(self.colors)] for i in range(len(drawdown_data))])
                ax.set_title('Maximum Drawdown by Strategy',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('Max Drawdown [%]', fontsize=10)
                ax.set_yticks(range(len(drawdown_data)))
                ax.set_yticklabels(drawdown_data.index)

                # Add value labels
                for bar, value in zip(bars, drawdown_data.values):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{value:.1f}%', ha='left', va='center', fontsize=9)
            plot_idx += 1

        # Drawdown Duration comparison
        if 'Max. Drawdown Duration' in available_cols:
            ax = axes[plot_idx] if len(available_cols) > 1 else axes[0]
            duration_data = df_clean.groupby(
                'Strategy')['Max. Drawdown Duration'].mean().dropna().sort_values()

            if not duration_data.empty:
                bars = ax.barh(range(len(duration_data)), duration_data.values,
                               color=[self.colors[i % len(self.colors)] for i in range(len(duration_data))])
                ax.set_title('Max Drawdown Duration by Strategy',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('Duration (Days)', fontsize=10)
                ax.set_yticks(range(len(duration_data)))
                ax.set_yticklabels(duration_data.index)

                # Add value labels
                for bar, value in zip(bars, duration_data.values):
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
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

        # Use core metric cleaning
        df_clean = self._get_cleaned_data(
            df, ['Return [%]'], strategy_specific=False)

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
        # Separate metrics by importance for different strategy types
        core_metrics = ['Return [%]', 'Sharpe Ratio',
                        'Max. Drawdown [%]', 'Volatility (Ann.) [%]']
        trading_metrics = ['Win Rate [%]', 'Profit Factor', '# Trades']

        # Filter metrics that exist in the dataframe
        available_core = [m for m in core_metrics if m in df.columns]
        available_trading = [m for m in trading_metrics if m in df.columns]

        if not available_core:
            print("None of the core metrics are available in the data")
            return None

        # Use flexible data cleaning
        df_clean = self._get_cleaned_data(
            df, available_core, strategy_specific=False)

        if df_clean.empty:
            print("No data available for summary table after cleaning")
            return None

        # Create summary for core metrics (all strategies)
        all_metrics = available_core + available_trading
        summary = df_clean.groupby('Strategy')[all_metrics].agg(
            ['mean', 'std', 'count']).round(3)

        # Flatten column names
        summary.columns = [
            f'{metric}_{stat}' for metric, stat in summary.columns]

        # Add ranking for each core metric (only rank where we have sufficient data)
        for metric in available_core:
            count_col = f'{metric}_count'
            if count_col in summary.columns:
                # Only rank strategies that have data for this metric
                valid_mask = summary[count_col] > 0
                if 'Drawdown' in metric:  # Lower is better for drawdown
                    summary.loc[valid_mask,
                                f'{metric}_rank'] = summary.loc[valid_mask, f'{metric}_mean'].rank()
                else:  # Higher is better for other metrics
                    summary.loc[valid_mask, f'{metric}_rank'] = summary.loc[valid_mask, f'{metric}_mean'].rank(
                        ascending=False)

        # Calculate overall score (average rank) only for core metrics
        rank_columns = [col for col in summary.columns if '_rank' in col and any(
            core in col for core in available_core)]
        if rank_columns:
            summary['Overall_Rank'] = summary[rank_columns].mean(axis=1)
            summary['Overall_Score'] = summary['Overall_Rank'].rank()

        print("COMPREHENSIVE STRATEGY COMPARISON SUMMARY")
        print("="*60)
        print("Note: Buy & Hold may show NaN for trading-specific metrics")
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
                combined_fig, individual_figs = viz.create_metric_comparison_bars(
                    df, save_path="metric_comparison.png")
                print(
                    f"Created combined metric comparison and {len(individual_figs)} individual plots")
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
