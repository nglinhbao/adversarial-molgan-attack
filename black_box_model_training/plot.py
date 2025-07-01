import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

def load_all_results(models_dir="models"):
    """Load all CSV result files and combine them."""
    models_path = Path(models_dir)
    csv_files = list(models_path.glob("*_results.csv"))
    
    all_results = []
    for csv_file in csv_files:
        dataset_name = csv_file.stem.replace("_results", "")
        df = pd.read_csv(csv_file)
        df['Dataset'] = dataset_name.upper()
        all_results.append(df)
    
    if not all_results:
        print("No CSV result files found!")
        return None
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Clean up the data
    combined_df = combined_df[combined_df['Score'] != 'skipped']
    combined_df['Score'] = pd.to_numeric(combined_df['Score'])
    
    return combined_df

def normalize_scores(df):
    """Normalize scores within each metric for ranking purposes."""
    df_normalized = df.copy()
    
    for metric in df['Metric'].unique():
        metric_mask = df['Metric'] == metric
        metric_data = df[metric_mask]
        
        # Check if higher is better
        higher_better = metric_data['Direction'].iloc[0] == '↑'
        
        if higher_better:
            # For metrics where higher is better (ROC_AUC), normalize to 0-1
            min_score = metric_data['Score'].min()
            max_score = metric_data['Score'].max()
            df_normalized.loc[metric_mask, 'Normalized_Score'] = (
                (metric_data['Score'] - min_score) / (max_score - min_score)
            )
        else:
            # For metrics where lower is better (MAE), invert and normalize
            min_score = metric_data['Score'].min()
            max_score = metric_data['Score'].max()
            df_normalized.loc[metric_mask, 'Normalized_Score'] = (
                (max_score - metric_data['Score']) / (max_score - min_score)
            )
    
    return df_normalized

def plot_model_performance(df):
    """Create a comprehensive plot of model performance across datasets."""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model Performance Across Datasets and Metrics', fontsize=16, fontweight='bold')
    
    # 1. Separate heatmaps for each metric
    ax1 = axes[0, 0]
    metrics = df['Metric'].unique()
    
    if len(metrics) == 1:
        # Single metric heatmap
        pivot_df = df.pivot(index='Model', columns='Dataset', values='Score')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, 
                   cbar_kws={'label': f'Score ({metrics[0]})'})
        ax1.set_title(f'Performance Heatmap - {metrics[0]}')
    else:
        # Multiple metrics - create subplot heatmaps
        ax1.remove()
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        
        for i, metric in enumerate(metrics):
            metric_df = df[df['Metric'] == metric]
            pivot_df = metric_df.pivot(index='Model', columns='Dataset', values='Score')
            
            ax_heat = fig.add_subplot(gs[0, i])
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax_heat,
                       cbar_kws={'label': f'Score'})
            ax_heat.set_title(f'{metric} Performance')
            ax_heat.set_xlabel('Dataset')
            if i == 0:
                ax_heat.set_ylabel('Model')
    
    # 2. Grouped bar plot by dataset and metric
    ax2 = axes[0, 1] if len(metrics) == 1 else fig.add_subplot(gs[1, :])
    
    datasets = df['Dataset'].unique()
    models = df['Model'].unique()
    
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for metric in metrics:
        metric_df = df[df['Metric'] == metric]
        
        for i, model in enumerate(models):
            model_scores = []
            for dataset in datasets:
                score = metric_df[(metric_df['Dataset'] == dataset) & 
                                (metric_df['Model'] == model)]['Score']
                if not score.empty:
                    model_scores.append(score.iloc[0])
                else:
                    model_scores.append(np.nan)
            
            offset = x + i * width - width * (len(models) - 1) / 2
            bars = ax2.bar(offset, model_scores, width, label=f'{model}' if len(metrics) == 1 else f'{model}-{metric}', 
                          color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Performance by Dataset' + (f' and Metric' if len(metrics) > 1 else ''))
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot separated by metric
    if len(metrics) == 1:
        ax3 = axes[1, 0]
        sns.boxplot(data=df, x='Model', y='Score', ax=ax3)
        ax3.set_title(f'Score Distribution by Model ({metrics[0]})')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    else:
        ax3 = axes[1, 0]
        ax3.remove()
        for i, metric in enumerate(metrics):
            metric_df = df[df['Metric'] == metric]
            ax_box = fig.add_subplot(2, 2, 3 + i)
            sns.boxplot(data=metric_df, x='Model', y='Score', ax=ax_box)
            ax_box.set_title(f'{metric} Distribution')
            ax_box.tick_params(axis='x', rotation=45)
            ax_box.grid(True, alpha=0.3)
    
    # 4. Model ranking using normalized scores
    if len(metrics) == 1:
        ax4 = axes[1, 1]
        model_avg = df.groupby('Model')['Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        # Check if higher is better for proper sorting
        higher_better = df['Direction'].iloc[0] == '↑'
        if not higher_better:
            model_avg = model_avg.sort_values('mean', ascending=True)
        
        bars = ax4.barh(range(len(model_avg)), model_avg['mean'], 
                       xerr=model_avg['std'], capsize=5, alpha=0.8, color='skyblue')
        ax4.set_yticks(range(len(model_avg)))
        ax4.set_yticklabels(model_avg.index)
        ax4.set_xlabel(f'Average Score ({metrics[0]})')
        ax4.set_title('Model Ranking (Average ± Std)')
        ax4.grid(True, alpha=0.3)
        
        for i, (mean_val, std_val) in enumerate(zip(model_avg['mean'], model_avg['std'])):
            ax4.text(mean_val + 0.01, i, f'{mean_val:.3f}±{std_val:.3f}', 
                    va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_dataset_specific(df, dataset_name):
    """Create a detailed plot for a specific dataset."""
    dataset_df = df[df['Dataset'] == dataset_name]
    
    if dataset_df.empty:
        print(f"No data found for dataset: {dataset_name}")
        return None
    
    metrics = dataset_df['Metric'].unique()
    
    if len(metrics) == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
    
    fig.suptitle(f'{dataset_name} Dataset - Model Performance', fontsize=14, fontweight='bold')
    
    for metric_idx, metric in enumerate(metrics):
        metric_df = dataset_df[dataset_df['Metric'] == metric]
        
        # Determine if higher is better
        higher_better = metric_df['Direction'].iloc[0] == '↑'
        sorted_df = metric_df.sort_values('Score', ascending=not higher_better)
        
        if len(metrics) == 1:
            ax1, ax2 = fig.axes
        else:
            ax1, ax2 = axes[metric_idx*2], axes[metric_idx*2 + 1]
        
        # Bar plot
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df)))
        
        bars1 = ax1.bar(range(len(sorted_df)), sorted_df['Score'], color=colors, alpha=0.8)
        ax1.set_xticks(range(len(sorted_df)))
        ax1.set_xticklabels(sorted_df['Model'], rotation=45, ha='right')
        ax1.set_ylabel(f"Score ({metric})")
        ax1.set_title(f'{metric} Performance ({"↑ Higher Better" if higher_better else "↓ Lower Better"})')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, sorted_df['Score']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Radar chart
        angles = np.linspace(0, 2*np.pi, len(sorted_df), endpoint=False).tolist()
        angles += angles[:1]
        
        scores = sorted_df['Score'].tolist()
        scores += scores[:1]
        
        if len(metrics) == 1:
            ax2 = plt.subplot(122, projection='polar')
        else:
            ax2 = plt.subplot(2, 2, metric_idx*2 + 2, projection='polar')
        
        ax2.plot(angles, scores, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax2.fill(angles, scores, alpha=0.25, color='blue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(sorted_df['Model'])
        ax2.set_title(f'{metric} Radar Chart', pad=20)
        ax2.grid(True)
    
    plt.tight_layout()
    return fig

def create_summary_table(df):
    """Create a summary table of results separated by metric."""
    print("\n" + "="*90)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*90)
    
    # Summary by metric
    for metric in df['Metric'].unique():
        metric_df = df[df['Metric'] == metric]
        higher_better = metric_df['Direction'].iloc[0] == '↑'
        
        print(f"\n{metric} METRIC ({'Higher is Better' if higher_better else 'Lower is Better'})")
        print("-" * 70)
        
        model_stats = metric_df.groupby('Model')['Score'].agg(['count', 'mean', 'std', 'min', 'max'])
        model_stats = model_stats.sort_values('mean', ascending=not higher_better)
        
        print(f"{'Model':<12} {'Datasets':<9} {'Avg Score':<10} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 65)
        
        for model, stats in model_stats.iterrows():
            print(f"{model:<12} {stats['count']:<9.0f} {stats['mean']:<10.3f} "
                  f"{stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f}")
    
    # Dataset-wise best models
    print(f"\n{'Dataset':<10} {'Metric':<10} {'Best Model':<15} {'Score':<10}")
    print("-" * 50)
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        
        for metric in dataset_df['Metric'].unique():
            metric_dataset_df = dataset_df[dataset_df['Metric'] == metric]
            higher_better = metric_dataset_df['Direction'].iloc[0] == '↑'
            
            if higher_better:
                best_row = metric_dataset_df.loc[metric_dataset_df['Score'].idxmax()]
            else:
                best_row = metric_dataset_df.loc[metric_dataset_df['Score'].idxmin()]
            
            print(f"{dataset:<10} {metric:<10} {best_row['Model']:<15} {best_row['Score']:<10.3f}")

def create_cross_metric_ranking(df):
    """Create a ranking that considers performance across different metrics."""
    print(f"\n{'='*80}")
    print("CROSS-METRIC MODEL RANKING (Based on Normalized Scores)")
    print("="*80)
    
    df_norm = normalize_scores(df)
    
    # Calculate average normalized score for each model
    model_ranking = df_norm.groupby('Model')['Normalized_Score'].agg(['mean', 'std', 'count'])
    model_ranking = model_ranking.sort_values('mean', ascending=False)
    
    print(f"{'Rank':<5} {'Model':<12} {'Avg Norm Score':<15} {'Std':<8} {'Datasets':<9}")
    print("-" * 55)
    
    for rank, (model, stats) in enumerate(model_ranking.iterrows(), 1):
        print(f"{rank:<5} {model:<12} {stats['mean']:<15.3f} {stats['std']:<8.3f} {stats['count']:<9.0f}")
    
    return model_ranking

def main():
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    print(f"Created plots directory: {plots_dir}")
    
    # Load all results
    df = load_all_results()
    
    if df is None:
        return
    
    print(f"Loaded results for {len(df)} model-dataset combinations")
    print(f"Datasets: {', '.join(df['Dataset'].unique())}")
    print(f"Models: {', '.join(df['Model'].unique())}")
    print(f"Metrics: {', '.join(df['Metric'].unique())}")
    
    # Create comprehensive plot
    print("\nCreating comprehensive performance plots...")
    fig1 = plot_model_performance(df)
    comprehensive_path = plots_dir / 'model_performance_comprehensive.png'
    fig1.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {comprehensive_path}")
    
    # Create dataset-specific plots
    for dataset in df['Dataset'].unique():
        print(f"\nCreating plot for {dataset} dataset...")
        fig2 = plot_dataset_specific(df, dataset)
        if fig2:
            dataset_path = plots_dir / f'{dataset.lower()}_performance.png'
            fig2.savefig(dataset_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {dataset_path}")
    
    # Save summary table to file
    summary_path = plots_dir / 'performance_summary.txt'
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    create_summary_table(df)
    cross_metric_ranking = create_cross_metric_ranking(df)
    
    # Restore stdout and save to file
    sys.stdout = old_stdout
    summary_text = captured_output.getvalue()
    
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\nSaved summary table: {summary_path}")
    
    # Also print to console
    print(summary_text)
    
    print(f"\nAll plots and summary saved to: {plots_dir}/")
    print("Files created:")
    for file in sorted(plots_dir.glob("*")):
        print(f"  - {file.name}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()