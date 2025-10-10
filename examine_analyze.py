#!/usr/bin/env python3
"""
Examine and improve the analyze.py script
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def examine_current_analyze_script():
    """Examine the current analyze.py script."""
    
    print("🔍 Examining current analyze.py script...")
    
    analyze_file = Path("analyze.py")
    if not analyze_file.exists():
        print("❌ analyze.py not found!")
        return None
    
    with open(analyze_file, 'r') as f:
        content = f.read()
    
    print(f"📄 Current analyze.py content:")
    print("=" * 50)
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}: {line}")
    print("=" * 50)
    
    return content

def find_log_files():
    """Find all log directories and files from your runs."""
    
    print(f"\n🔍 Finding log files from your runs...")
    
    # Look for log directories
    log_dirs = []
    for item in Path(".").iterdir():
        if item.is_dir() and ("log" in item.name.lower() or "final_logs" in item.name):
            log_dirs.append(item)
    
    print(f"📁 Found log directories:")
    for log_dir in log_dirs:
        print(f"  📂 {log_dir}")
        
        # Look for prediction files or results
        for subdir in log_dir.rglob("*"):
            if subdir.is_dir():
                print(f"    📂 {subdir.name}")
                for file in subdir.iterdir():
                    if file.is_file():
                        print(f"      📄 {file.name} ({file.stat().st_size} bytes)")
    
    return log_dirs

def analyze_your_results():
    """Analyze your BioDiscoveryAgent results."""
    
    print(f"\n📊 Analyzing your BioDiscoveryAgent results...")
    
    # Look for the most recent final_logs
    final_log_dir = None
    for item in Path(".").iterdir():
        if item.is_dir() and "final_logs" in item.name:
            final_log_dir = item
            break
    
    if not final_log_dir:
        print("❌ No final_logs directory found")
        return
    
    print(f"📁 Analyzing: {final_log_dir}")
    
    # Look for prediction files
    prediction_files = []
    for file in final_log_dir.rglob("*.npy"):
        if "pred" in file.name.lower():
            prediction_files.append(file)
    
    if prediction_files:
        print(f"📄 Found prediction files:")
        for pred_file in prediction_files:
            print(f"  {pred_file}")
            
            try:
                predictions = np.load(pred_file, allow_pickle=True)
                print(f"    Shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
                print(f"    Sample: {predictions[:5] if len(predictions) > 5 else predictions}")
            except Exception as e:
                print(f"    Error loading: {e}")
    
    # Look for log files with gene predictions
    log_files = list(final_log_dir.rglob("*.log"))
    if log_files:
        print(f"\n📄 Analyzing log files:")
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"📄 Latest log: {latest_log}")
        
        # Extract predicted genes from log
        predicted_genes = extract_genes_from_log(latest_log)
        if predicted_genes:
            print(f"🎯 Genes discovered in this run:")
            for step, genes in predicted_genes.items():
                print(f"  Step {step}: {len(genes)} genes")
                print(f"    {genes[:10]}{'...' if len(genes) > 10 else ''}")

def extract_genes_from_log(log_file):
    """Extract predicted genes from log file."""
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for gene predictions in the log
        predicted_genes = {}
        
        # Split by steps
        step_sections = content.split("Step ")
        
        for i, section in enumerate(step_sections[1:], 1):  # Skip first empty section
            genes_in_step = []
            
            # Look for GENE_ patterns
            import re
            gene_matches = re.findall(r'GENE_\d+', section)
            genes_in_step.extend(gene_matches)
            
            if genes_in_step:
                predicted_genes[i] = list(set(genes_in_step))  # Remove duplicates
        
        return predicted_genes
        
    except Exception as e:
        print(f"❌ Error extracting genes from log: {e}")
        return {}

def evaluate_performance():
    """Evaluate how well the predictions performed."""
    
    print(f"\n📊 Evaluating prediction performance...")
    
    # Load your ground truth
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    if not gt_file.exists():
        print("❌ Ground truth file not found")
        return
    
    gt_df = pd.read_csv(gt_file)
    true_positives = set(gt_df[gt_df['Label'] == 1]['Gene'])
    true_negatives = set(gt_df[gt_df['Label'] == 0]['Gene'])
    
    print(f"📊 Ground truth:")
    print(f"  Positive genes (hits): {len(true_positives)}")
    print(f"  Negative genes: {len(true_negatives)}")
    print(f"  Sample positives: {list(true_positives)[:5]}")
    
    # Load your dataset scores
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Get top scoring genes for comparison
    top_genes = set(df.nlargest(len(true_positives), df.columns[0]).index)
    
    print(f"\n📊 Top scoring genes (baseline):")
    print(f"  Count: {len(top_genes)}")
    print(f"  Sample: {list(top_genes)[:5]}")
    
    # Compare with ground truth
    baseline_overlap = len(top_genes.intersection(true_positives))
    baseline_precision = baseline_overlap / len(top_genes)
    baseline_recall = baseline_overlap / len(true_positives)
    
    print(f"\n📊 Baseline performance (top scoring genes):")
    print(f"  Overlap with true positives: {baseline_overlap}/{len(top_genes)}")
    print(f"  Precision: {baseline_precision:.3f}")
    print(f"  Recall: {baseline_recall:.3f}")
    
    return {
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'top_genes': top_genes,
        'baseline_precision': baseline_precision,
        'baseline_recall': baseline_recall
    }

if __name__ == "__main__":
    print("🔧 Examining and Improving analyze.py")
    print("=" * 50)
    
    # Step 1: Examine current analyze.py
    current_content = examine_current_analyze_script()
    
    # Step 2: Find your log files
    log_dirs = find_log_files()
    
    # Step 3: Analyze your results
    analyze_your_results()
    
    # Step 4: Evaluate performance
    performance = evaluate_performance()
    
    print(f"\n💡 Next steps:")
    print(f"1. Run the improved analyze script I'll create")
    print(f"2. Compare BioDiscoveryAgent predictions vs. baseline")
    print(f"3. Identify which genes were successfully discovered")
    print(f"4. Look for patterns in successful predictions")
