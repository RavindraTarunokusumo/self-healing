#!/usr/bin/env python3
"""
Visualization Dashboard for Self-Healing Agent Comparison Study

Creates a comprehensive 4-panel dashboard displaying key metrics:
- Panel 1: Pass Rate with Confidence Intervals
- Panel 2: Cost Efficiency Scatter Plot with Pareto Frontier
- Panel 3: Self-Healing Impact (Zero-Shot + Lift)
- Panel 4: Cost and Iteration Analysis

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --input results/comparison.csv --output results/dashboard.png
    python scripts/visualize_results.py --format both  # PNG + PDF
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Semantic color scheme based on model capability tiers
MODEL_COLORS = {
    'max-max': '#6B46C1',      # Deep Purple (premium, most expensive)
    'flash-max': '#2563EB',    # Royal Blue (hybrid premium)
    'coder-coder': '#10B981',  # Emerald Green (specialized mid-tier)
    'flash-coder': '#059669',  # Darker Emerald (specialized mid-tier variant)
    'flash-flash': '#F59E0B',  # Amber (budget option)
}

# Lighter shades for base bars in stacked charts
MODEL_COLORS_LIGHT = {
    'max-max': '#A78BFA',      # Light Purple
    'flash-max': '#93C5FD',    # Light Blue
    'coder-coder': '#6EE7B7',  # Light Emerald
    'flash-coder': '#6EE7B7',  # Light Emerald
    'flash-flash': '#FCD34D',  # Light Amber
}

# Display names for cleaner labels
CONFIG_DISPLAY_NAMES = {
    'max-max': 'Max-Max',
    'flash-max': 'Flash-Max',
    'coder-coder': 'Coder-Coder',
    'flash-coder': 'Flash-Coder',
    'flash-flash': 'Flash-Flash',
}


def load_comparison_data(csv_path: str) -> pd.DataFrame:
    """Load comparison data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Add display names
    df['display_name'] = df['config'].map(CONFIG_DISPLAY_NAMES)

    return df


def get_color(config: str) -> str:
    """Get color for a configuration."""
    return MODEL_COLORS.get(config, '#6B7280')  # Default gray


def get_light_color(config: str) -> str:
    """Get light color for a configuration."""
    return MODEL_COLORS_LIGHT.get(config, '#D1D5DB')  # Default light gray


def plot_pass_rate_panel(ax: plt.Axes, data: pd.DataFrame):
    """Panel 1: Pass Rate with 95% Confidence Intervals (Horizontal Bar Chart)."""
    # Sort by pass rate descending
    sorted_data = data.sort_values('pass_rate_mean', ascending=True)

    y_pos = np.arange(len(sorted_data))
    colors = [get_color(c) for c in sorted_data['config']]

    # Calculate error bar sizes (asymmetric based on CI)
    xerr_low = sorted_data['pass_rate_mean'] - sorted_data['pass_rate_ci_low']
    xerr_high = sorted_data['pass_rate_ci_high'] - sorted_data['pass_rate_mean']

    # Create horizontal bars
    ax.barh(y_pos, sorted_data['pass_rate_mean'], color=colors,
            xerr=[xerr_low, xerr_high], capsize=4,
            error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': '#374151'})

    # Add value annotations
    for i, (_, row) in enumerate(sorted_data.iterrows()):
        ax.annotate(f"{row['pass_rate_mean']:.1f}%",
                   xy=(row['pass_rate_mean'] + 0.5, i),
                   va='center', ha='left', fontsize=9, fontweight='bold')

    # Add 95% excellence threshold line
    ax.axvline(x=95, color='#DC2626', linestyle='--', linewidth=1.5, alpha=0.7, label='95% threshold')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CONFIG_DISPLAY_NAMES.get(c, c) for c in sorted_data['config']])
    ax.set_xlabel('Pass Rate (%)', fontsize=11)
    ax.set_title('Pass Rate with 95% CI', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlim(88, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=8)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def identify_pareto_frontier(costs: np.ndarray, pass_rates: np.ndarray) -> list:
    """
    Identify Pareto optimal points.
    Pareto optimal = no other point has both higher pass rate AND lower cost.
    """
    n = len(costs)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if j has higher pass rate AND lower cost
                if pass_rates[j] >= pass_rates[i] and costs[j] <= costs[i]:
                    if pass_rates[j] > pass_rates[i] or costs[j] < costs[i]:
                        pareto_mask[i] = False
                        break

    return list(np.where(pareto_mask)[0])


def plot_cost_efficiency_panel(ax: plt.Axes, data: pd.DataFrame):
    """Panel 2: Cost Efficiency Scatter Plot with Pareto Frontier."""
    # Scale point sizes by total cost (normalized)
    max_cost = data['total_cost_mean'].max()
    min_cost = data['total_cost_mean'].min()
    size_scale = 100 + 400 * (data['total_cost_mean'] - min_cost) / (max_cost - min_cost)

    colors = [get_color(c) for c in data['config']]

    # Scatter plot
    ax.scatter(data['pass_rate_mean'], data['cost_per_pass_mean'] * 1000,  # Convert to milli-dollars
               s=size_scale, c=colors, alpha=0.8, edgecolors='white', linewidths=2)

    # Add labels near each point
    for _, row in data.iterrows():
        # Offset based on position to avoid overlap
        x_offset = 0.3 if row['pass_rate_mean'] < 95 else -0.3
        ha = 'left' if x_offset > 0 else 'right'
        ax.annotate(CONFIG_DISPLAY_NAMES.get(row['config'], row['config']),
                   xy=(row['pass_rate_mean'], row['cost_per_pass_mean'] * 1000),
                   xytext=(x_offset, 0.1), textcoords='offset points',
                   fontsize=8, ha=ha, va='bottom', fontweight='bold')

    # Identify and draw Pareto frontier
    costs = data['cost_per_pass_mean'].values * 1000
    pass_rates = data['pass_rate_mean'].values
    pareto_indices = identify_pareto_frontier(costs, pass_rates)

    if len(pareto_indices) > 1:
        # Sort Pareto points by pass rate for line drawing
        pareto_points = [(pass_rates[i], costs[i]) for i in pareto_indices]
        pareto_points.sort(key=lambda x: x[0])

        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'k--', linewidth=1.5, alpha=0.5, label='Pareto frontier')

    # Add quadrant annotations
    ax.annotate('High Performance\nLow Cost', xy=(96.5, 0.3), fontsize=7,
               color='#059669', ha='center', style='italic', alpha=0.8)
    ax.annotate('High Performance\nHigh Cost', xy=(96.5, 2.2), fontsize=7,
               color='#6B46C1', ha='center', style='italic', alpha=0.8)

    # Styling
    ax.set_xlabel('Pass Rate (%)', fontsize=11)
    ax.set_ylabel('Cost per Pass (m$)', fontsize=11)
    ax.set_title('Cost Efficiency Trade-off', fontsize=14, fontweight='bold', pad=10)
    ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=8)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:.1f}m'))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_self_healing_impact_panel(ax: plt.Axes, data: pd.DataFrame):
    """Panel 3: Self-Healing Impact Stacked Bars (Zero-Shot + Lift)."""
    # Sort by total pass rate for consistency
    sorted_data = data.sort_values('pass_rate_mean', ascending=False)

    x_pos = np.arange(len(sorted_data))
    bar_width = 0.6

    # Base bars (zero-shot)
    light_colors = [get_light_color(c) for c in sorted_data['config']]
    ax.bar(x_pos, sorted_data['zero_shot_mean'], bar_width,
           color=light_colors, label='Zero-Shot Rate', edgecolor='white', linewidth=0.5)

    # Lift bars (stacked on top)
    dark_colors = [get_color(c) for c in sorted_data['config']]
    ax.bar(x_pos, sorted_data['self_heal_lift_mean'], bar_width,
           bottom=sorted_data['zero_shot_mean'], color=dark_colors,
           label='Self-Heal Lift', edgecolor='white', linewidth=0.5,
           hatch='///', alpha=0.9)

    # Add lift annotations on the lift bars
    for i, (_, row) in enumerate(sorted_data.iterrows()):
        lift_y = row['zero_shot_mean'] + row['self_heal_lift_mean'] / 2
        ax.annotate(f"+{row['self_heal_lift_mean']:.1f}%",
                   xy=(i, lift_y), ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')

        # Add total at top
        total_y = row['zero_shot_mean'] + row['self_heal_lift_mean']
        ax.annotate(f"{row['pass_rate_mean']:.1f}%",
                   xy=(i, total_y + 1), ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='#374151')

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONFIG_DISPLAY_NAMES.get(c, c) for c in sorted_data['config']],
                       rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Pass Rate (%)', fontsize=11)
    ax.set_title('Self-Healing Impact Analysis', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Add insight annotation
    ax.annotate('Cheaper models benefit more from self-healing',
               xy=(0.5, 0.02), xycoords='axes fraction',
               fontsize=8, style='italic', color='#6B7280', ha='center')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_cost_breakdown_panel(ax: plt.Axes, data: pd.DataFrame):
    """Panel 4: Total Cost with Iteration Annotations."""
    # Sort by total cost for visual clarity
    sorted_data = data.sort_values('total_cost_mean', ascending=False)

    x_pos = np.arange(len(sorted_data))
    bar_width = 0.6

    colors = [get_color(c) for c in sorted_data['config']]

    ax.bar(x_pos, sorted_data['total_cost_mean'], bar_width, color=colors,
           edgecolor='white', linewidth=0.5)

    # Add iteration annotations on each bar
    for i, (_, row) in enumerate(sorted_data.iterrows()):
        # Cost annotation at top
        ax.annotate(f"${row['total_cost_mean']:.3f}",
                   xy=(i, row['total_cost_mean'] + 0.01), ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    # Add cost ratio annotation
    max_cost = sorted_data['total_cost_mean'].max()
    min_cost = sorted_data['total_cost_mean'].min()
    ratio = max_cost / min_cost
    ax.annotate(f'{ratio:.0f}x cost difference',
               xy=(0.5, 0.95), xycoords='axes fraction',
               fontsize=10, fontweight='bold', color='#DC2626', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor='#DC2626', alpha=0.8))

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels([CONFIG_DISPLAY_NAMES.get(c, c) for c in sorted_data['config']],
                       rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Total Cost ($)', fontsize=11)
    ax.set_title('Total Cost', fontsize=14, fontweight='bold', pad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:.2f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_dashboard(data: pd.DataFrame) -> plt.Figure:
    """Create the 4-panel dashboard visualization."""
    # Set up the figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=100)

    # Set overall style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Configure fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

    # Panel 1: Pass Rate (Top-Left)
    plot_pass_rate_panel(axes[0, 0], data)

    # Panel 2: Cost Efficiency (Top-Right)
    plot_cost_efficiency_panel(axes[0, 1], data)

    # Panel 3: Self-Healing Impact (Bottom-Left)
    plot_self_healing_impact_panel(axes[1, 0], data)

    # Panel 4: Cost Breakdown (Bottom-Right)
    plot_cost_breakdown_panel(axes[1, 1], data)

    # Add main title
    fig.suptitle('Self-Healing Agent Comparison Dashboard',
                fontsize=18, fontweight='bold', y=0.98)

    # Add subtitle with key insight
    fig.text(0.5, 0.94,
            'HumanEval Benchmark: 5 runs per configuration, 164 problems each',
            ha='center', fontsize=11, style='italic', color='#6B7280')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.93], h_pad=3, w_pad=3)

    # Add footer with key findings
    footer_text = (
        'Key Findings: Flash-Flash achieves 95% pass rate at 19x lower cost ($0.00013/pass vs $0.0024/pass). '
        'Self-healing provides +9.6% to +17.6% improvement, with cheaper models benefiting most.'
    )
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=9, style='italic',
            color='#4B5563', wrap=True)

    return fig


def save_dashboard(fig: plt.Figure, output_path: str, format_type: str = 'png'):
    """Save the dashboard figure to file."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save based on format
    if format_type == 'both':
        # Save PNG
        png_path = output_path.replace('.pdf', '.png')
        if not png_path.endswith('.png'):
            png_path = output_path.rsplit('.', 1)[0] + '.png'
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved PNG: {png_path}")

        # Save PDF
        pdf_path = output_path.replace('.png', '.pdf')
        if not pdf_path.endswith('.pdf'):
            pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved PDF: {pdf_path}")
    else:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization dashboard for self-healing agent comparison study'
    )
    parser.add_argument('--input', '-i', default='results/comparison.csv',
                       help='Path to comparison CSV file (default: results/comparison.csv)')
    parser.add_argument('--output', '-o', default='results/comparison_dashboard.png',
                       help='Output file path (default: results/comparison_dashboard.png)')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'both'], default='png',
                       help='Output format (default: png)')
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")

    try:
        # Load data
        data = load_comparison_data(args.input)
        print(f"Loaded {len(data)} configurations")

        # Create dashboard
        print("Creating 4-panel dashboard...")
        fig = create_dashboard(data)

        # Save output
        save_dashboard(fig, args.output, args.format)

        # Close figure to free memory
        plt.close(fig)

        print("\nVisualization complete!")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
