#!/usr/bin/env python3
"""
Quick fix for the analysis error with small datasets
"""

import pandas as pd
import numpy as np

def fix_improved_analyze():
    """Fix the improved_analyze.py script to handle small datasets."""
    
    print("🔧 Fixing improved_analyze.py for small datasets...")
    
    # Read the current script
    with open('improved_analyze.py', 'r') as f:
        content = f.read()
    
    # Find and replace the problematic section
    old_code = """    # Baseline 2: Random selection
    random_genes = set(np.random.choice(df.index, num_predictions, replace=False))
    baselines['random'] = calculate_metrics(random_genes, true_positives, set())"""
    
    new_code = """    # Baseline 2: Random selection (handle small datasets)
    max_sample_size = min(num_predictions, len(df))
    random_genes = set(np.random.choice(df.index, max_sample_size, replace=False))
    baselines['random'] = calculate_metrics(random_genes, true_positives, set())"""
    
    # Replace the code
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write the fixed version
        with open('improved_analyze_fixed.py', 'w') as f:
            f.write(content)
        
        print("✅ Created improved_analyze_fixed.py with small dataset support")
        return True
    else:
        print("❌ Could not find the exact code to replace")
        return False

def analyze_easy_results_manually():
    """Manually analyze the easy dataset results."""
    
    print("\n📊 Manual analysis of EASY_SCREEN results...")
    
    # The easy dataset has 100 genes, all positive
    # AI predicted 199 genes, but only 100 exist in dataset
    # This means AI found all 100 genes plus tried to predict 99 more
    
    print("🔍 Easy dataset analysis:")
    print("   Dataset size: 100 genes (all positive)")
    print("   AI predictions: 199 attempts")
    print("   Result: AI likely found most/all of the 100 genes")
    print("   Expected precision: ~50% (100 real genes / 199 predictions)")
    print("   Expected recall: ~100% (found all available genes)")
    
    # Try to load and analyze the actual results
    try:
        # Check if we can extract the actual gene predictions
        log_file = "easy_logs_EASY_SCREEN/easy_test/step_0_log.log"
        
        if Path(log_file).exists():
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract GENE_ patterns
            import re
            predicted_genes = re.findall(r'GENE_\d+', content)
            unique_predictions = list(set(predicted_genes))
            
            print(f"\n📄 Extracted from log:")
            print(f"   Raw predictions found: {len(predicted_genes)}")
            print(f"   Unique predictions: {len(unique_predictions)}")
            print(f"   Sample predictions: {unique_predictions[:10]}")
            
            # Load the easy dataset to see overlap
            easy_file = "data/EASY_SCREEN/EASY_SCREEN.csv"
            if Path(easy_file).exists():
                df = pd.read_csv(easy_file, index_col=0)
                dataset_genes = set(df.index)
                
                # Find overlap
                valid_predictions = set(unique_predictions).intersection(dataset_genes)
                invalid_predictions = set(unique_predictions) - dataset_genes
                
                print(f"\n🎯 Overlap analysis:")
                print(f"   Dataset genes: {len(dataset_genes)}")
                print(f"   Valid predictions: {len(valid_predictions)}")
                print(f"   Invalid predictions: {len(invalid_predictions)}")
                print(f"   Precision: {len(valid_predictions)/len(unique_predictions)*100:.1f}%")
                print(f"   Recall: {len(valid_predictions)/len(dataset_genes)*100:.1f}%")
                
                return {
                    'dataset_size': len(dataset_genes),
                    'predictions': len(unique_predictions),
                    'valid_predictions': len(valid_predictions),
                    'precision': len(valid_predictions)/len(unique_predictions),
                    'recall': len(valid_predictions)/len(dataset_genes)
                }
        
    except Exception as e:
        print(f"❌ Error in manual analysis: {e}")
    
    return None

if __name__ == "__main__":
    print("🔧 Fixing Analysis Error")
    print("=" * 30)
    
    from pathlib import Path
    
    # Step 1: Fix the analysis script
    fixed = fix_improved_analyze()
    
    # Step 2: Manual analysis of easy results
    results = analyze_easy_results_manually()
    
    if fixed:
        print(f"\n🚀 Now try the fixed analysis:")
        print(f"python3 improved_analyze_fixed.py --data_name EASY_SCREEN --log_dir easy_logs_EASY_SCREEN/easy_test --output_dir easy_analysis_fixed")
    
    if results:
        print(f"\n📊 Easy dataset performance:")
        print(f"   Precision: {results['precision']*100:.1f}%")
        print(f"   Recall: {results['recall']*100:.1f}%")
        
        if results['precision'] > 0.8:
            print("✅ Excellent precision on easy dataset!")
        elif results['precision'] > 0.6:
            print("✅ Good precision on easy dataset!")
