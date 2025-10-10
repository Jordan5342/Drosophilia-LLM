#!/usr/bin/env python3
"""
Improved analyze.py script for evaluating BioDiscoveryAgent performance
"""

import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse

def load_ground_truth(data_name):
    """Load ground truth data."""
    
    gt_file = Path(f"data/{data_name}/{data_name}_ground_truth.csv")
    if not gt_file.exists():
        print(f"❌ Ground truth file not found: {gt_file}")
        return None, None, None
    
    gt_df = pd.read_csv(gt_file)
    true_positives = set(gt_df[gt_df['Label'] == 1]['Gene'])
    true_negatives = set(gt_df[gt_df['Label'] == 0]['Gene'])
    
    print(f"📊 Ground truth loaded:")
    print(f"  Positive genes: {len(true_positives)}")
    print(f"  Negative genes: {len(true_negatives)}")
    
    return gt_df, true_positives, true_negatives

def load_dataset_scores(data_name):
    """Load the main dataset with gene scores."""
    
    main_file = Path(f"data/{data_name}/{data_name}.csv")
    if not main_file.exists():
        print(f"❌ Main dataset file not found: {main_file}")
        return None
    
    df = pd.read_csv(main_file, index_col=0)
    print(f"📊 Dataset loaded: {df.shape[0]} genes")
    
    return df

def extract_predictions_from_logs(log_dir):
    """Extract gene predictions from BioDiscoveryAgent log files."""
    
    print(f"🔍 Extracting predictions from {log_dir}...")
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return {}
    
    predictions_by_step = {}
    all_predictions = set()
    
    # Look for log files
    log_files = list(log_path.rglob("*.log"))
    
    for log_file in log_files:
        print(f"📄 Processing: {log_file}")
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract step number from filename or content
            step_match = re.search(r'step_(\d+)', log_file.name)
            if step_match:
                step_num = int(step_match.group(1))
            else:
                step_num = 0
            
            # Find gene predictions in the log
            gene_matches = re.findall(r'GENE_\d+', content)
            unique_genes = list(set(gene_matches))
            
            if unique_genes:
                predictions_by_step[step_num] = unique_genes
                all_predictions.update(unique_genes)
                print(f"  Step {step_num}: {len(unique_genes)} genes predicted")
        
        except Exception as e:
            print(f"❌ Error processing {log_file}: {e}")
    
    print(f"✅ Total unique predictions: {len(all_predictions)}")
    return predictions_by_step, all_predictions

def calculate_metrics(predicted_genes, true_positives, true_negatives):
    """Calculate performance metrics."""
    
    predicted_set = set(predicted_genes)
    
    # True positives: correctly predicted positive genes
    tp = len(predicted_set.intersection(true_positives))
    
    # False positives: predicted positive but actually negative
    fp = len(predicted_set.intersection(true_negatives))
    
    # False negatives: true positive but not predicted
    fn = len(true_positives - predicted_set)
    
    # True negatives: correctly not predicted (harder to calculate exactly)
    # We'll approximate based on available data
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_predicted': len(predicted_set),
        'total_true_positives': len(true_positives)
    }
    
    return metrics

def compare_with_baselines(df, true_positives, num_predictions):
    """Compare with baseline methods."""
    
    print(f"\n📊 Comparing with baseline methods...")
    
    baselines = {}
    
    # Baseline 1: Top scoring genes
    top_genes = set(df.nlargest(num_predictions, df.columns[0]).index)
    baselines['top_scoring'] = calculate_metrics(top_genes, true_positives, set())
    
    # Baseline 2: Random selection
    random_genes = set(np.random.choice(df.index, num_predictions, replace=False))
    baselines['random'] = calculate_metrics(random_genes, true_positives, set())
    
    # Baseline 3: Bottom scoring (as negative control)
    bottom_genes = set(df.nsmallest(num_predictions, df.columns[0]).index)
    baselines['bottom_scoring'] = calculate_metrics(bottom_genes, true_positives, set())
    
    return baselines

def create_visualizations(metrics_dict, save_dir):
    """Create visualization plots."""
    
    print(f"📊 Creating visualizations...")
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 1. Performance comparison bar plot
    methods = list(metrics_dict.keys())
    precisions = [metrics_dict[m]['precision'] for m in methods]
    recalls = [metrics_dict[m]['recall'] for m in methods]
    f1_scores = [metrics_dict[m]['f1_score'] for m in methods]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision
    ax1.bar(methods, precisions, color='skyblue')
    ax1.set_title('Precision')
    ax1.set_ylabel('Precision')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Recall
    ax2.bar(methods, recalls, color='lightgreen')
    ax2.set_title('Recall')
    ax2.set_ylabel('Recall')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # F1 Score
    ax3.bar(methods, f1_scores, color='salmon')
    ax3.set_title('F1 Score')
    ax3.set_ylabel('F1 Score')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrix style visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    biodiscovery_metrics = metrics_dict.get('biodiscovery', {})
    
    if biodiscovery_metrics:
        categories = ['True Positives', 'False Positives', 'False Negatives']
        values = [
            biodiscovery_metrics['true_positives'],
            biodiscovery_metrics['false_positives'], 
            biodiscovery_metrics['false_negatives']
        ]
        colors = ['green', 'red', 'orange']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_title('BioDiscoveryAgent Prediction Breakdown')
        ax.set_ylabel('Number of Genes')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {save_path}")

def generate_report(metrics_dict, predictions_by_step, all_predictions, true_positives, save_dir):
    """Generate a comprehensive analysis report."""
    
    print(f"📄 Generating analysis report...")
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    report_file = save_path / 'analysis_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("BioDiscoveryAgent Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total genes predicted: {len(all_predictions)}\n")
        f.write(f"Prediction steps: {len(predictions_by_step)}\n")
        f.write(f"True positive genes available: {len(true_positives)}\n\n")
        
        # Step-by-step predictions
        f.write("PREDICTIONS BY STEP\n")
        f.write("-" * 20 + "\n")
        for step, genes in sorted(predictions_by_step.items()):
            f.write(f"Step {step}: {len(genes)} genes\n")
            f.write(f"  Genes: {', '.join(sorted(genes)[:10])}")
            if len(genes) > 10:
                f.write(f" ... and {len(genes)-10} more")
            f.write("\n\n")
        
        # Performance metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 20 + "\n")
        for method, metrics in metrics_dict.items():
            f.write(f"{method.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']:.3f}\n")
            f.write(f"  Recall: {metrics['recall']:.3f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.3f}\n")
            f.write(f"  True Positives: {metrics['true_positives']}\n")
            f.write(f"  False Positives: {metrics['false_positives']}\n")
            f.write(f"  False Negatives: {metrics['false_negatives']}\n\n")
        
        # Successfully discovered genes
        if 'biodiscovery' in metrics_dict:
            discovered_positives = set(all_predictions).intersection(true_positives)
            f.write("SUCCESSFULLY DISCOVERED POSITIVE GENES\n")
            f.write("-" * 40 + "\n")
            for gene in sorted(discovered_positives):
                f.write(f"  {gene}\n")
            f.write(f"\nTotal: {len(discovered_positives)} genes\n\n")
        
        # Missed important genes
        missed_positives = true_positives - set(all_predictions)
        f.write("MISSED IMPORTANT GENES\n")
        f.write("-" * 25 + "\n")
        for gene in sorted(list(missed_positives)[:20]):  # Show first 20
            f.write(f"  {gene}\n")
        if len(missed_positives) > 20:
            f.write(f"  ... and {len(missed_positives)-20} more\n")
        f.write(f"\nTotal missed: {len(missed_positives)} genes\n")
    
    print(f"✅ Report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze BioDiscoveryAgent results')
    parser.add_argument('--data_name', type=str, default='MY_CUSTOM_SCREEN', 
                       help='Name of the dataset')
    parser.add_argument('--log_dir', type=str, default='final_logs_MY_CUSTOM_SCREEN/final_test',
                       help='Path to log directory')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print(f"🔧 Analyzing BioDiscoveryAgent Results")
    print(f"Data: {args.data_name}")
    print(f"Logs: {args.log_dir}")
    print("=" * 50)
    
    # Load data
    gt_df, true_positives, true_negatives = load_ground_truth(args.data_name)
    if gt_df is None:
        return
    
    df = load_dataset_scores(args.data_name)
    if df is None:
        return
    
    # Extract predictions
    predictions_by_step, all_predictions = extract_predictions_from_logs(args.log_dir)
    if not all_predictions:
        print("❌ No predictions found in logs")
        return
    
    # Calculate metrics
    print(f"\n📊 Calculating performance metrics...")
    
    metrics_dict = {}
    
    # BioDiscoveryAgent metrics
    biodiscovery_metrics = calculate_metrics(all_predictions, true_positives, true_negatives)
    metrics_dict['biodiscovery'] = biodiscovery_metrics
    
    # Baseline comparisons
    baselines = compare_with_baselines(df, true_positives, len(all_predictions))
    metrics_dict.update(baselines)
    
    # Print results
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'Method':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    for method, metrics in metrics_dict.items():
        print(f"{method:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")
    
    # Create visualizations
    create_visualizations(metrics_dict, args.output_dir)
    
    # Generate report
    generate_report(metrics_dict, predictions_by_step, all_predictions, true_positives, args.output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
