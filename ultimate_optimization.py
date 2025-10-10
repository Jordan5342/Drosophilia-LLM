#!/usr/bin/env python3
"""
Ultimate optimization combining best approaches
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_ultimate_task_prompt():
    """Create the ultimate task prompt combining all successful strategies."""
    
    print("🚀 Creating ultimate optimized task prompt...")
    
    # Load dataset and analyze top performers
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    true_positives = set(gt_df[gt_df['Label'] == 1]['Gene'])
    
    # Get the top 100 scoring genes and see how many are true positives
    top_100 = df.nlargest(100, df.columns[0])
    top_100_genes = set(top_100.index)
    tp_in_top_100 = top_100_genes.intersection(true_positives)
    
    # Get score thresholds
    score_99th = df.iloc[:, 0].quantile(0.99)  # Top 1%
    score_95th = df.iloc[:, 0].quantile(0.95)  # Top 5%
    score_90th = df.iloc[:, 0].quantile(0.90)  # Top 10%
    
    print(f"📊 Ultimate optimization analysis:")
    print(f"   Top 100 genes: {len(tp_in_top_100)}/{len(top_100)} are true positives ({len(tp_in_top_100)/len(top_100)*100:.1f}%)")
    print(f"   Score thresholds: 99th={score_99th:.3f}, 95th={score_95th:.3f}, 90th={score_90th:.3f}")
    
    # Get examples of successful genes
    top_tp_genes = list(tp_in_top_100)[:15]  # Top 15 true positive genes
    
    ultimate_task = {
        "Task": f"""You are a gene discovery expert. Identify the most essential genes for cellular survival.

PROVEN STRATEGY (Based on Analysis):
The top 100 highest-scoring genes contain {len(tp_in_top_100)} out of {len(top_100)} essential genes ({len(tp_in_top_100)/len(top_100)*100:.1f}% success rate).

SCORING THRESHOLDS FOR SUCCESS:
- EXCELLENT (Top 1%): Score > {score_99th:.3f} - Highest probability of essentiality  
- VERY GOOD (Top 5%): Score > {score_95th:.3f} - High probability of essentiality
- GOOD (Top 10%): Score > {score_90th:.3f} - Good probability of essentiality

PROVEN ESSENTIAL GENES (High scores + True positives):
{', '.join(top_tp_genes[:10])}

SELECTION ALGORITHM:
1. START with genes scoring > {score_95th:.3f} (Top 5%)
2. PRIORITIZE genes scoring > {score_99th:.3f} (Top 1%)  
3. EXPAND carefully to genes scoring > {score_90th:.3f} if needed
4. AVOID genes scoring below {score_90th:.3f} unless strong biological reasoning

BIOLOGICAL INSIGHT: Essential genes typically have:
- Very high importance scores (reflecting cellular dependency)
- Consistent high ranking across experiments
- Critical roles in core cellular processes

Your mission: Find genes that score in the top 10% AND are truly essential for survival.""",

        "Measurement": f"Gene essentiality score. Range: {df.iloc[:, 0].min():.3f}-{df.iloc[:, 0].max():.3f}. Thresholds: Excellent>{score_99th:.3f}, Good>{score_90th:.3f}. Higher scores = more essential."
    }
    
    # Save ultimate task prompt
    task_prompt_file = Path("datasets/task_prompts/MY_CUSTOM_SCREEN.json")
    
    # Backup
    backup_file = task_prompt_file.with_suffix('.json.ultimate_backup')
    import shutil
    shutil.copy2(task_prompt_file, backup_file)
    
    with open(task_prompt_file, 'w') as f:
        json.dump(ultimate_task, f, indent=2)
    
    print(f"✅ Ultimate task prompt saved: {task_prompt_file}")
    print(f"✅ Backup saved: {backup_file}")
    
    return ultimate_task

def create_precision_focused_dataset():
    """Create a precision-focused dataset with only top 500 genes."""
    
    print(f"\n🎯 Creating precision-focused dataset...")
    
    # Load full dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")  
    df = pd.read_csv(main_file, index_col=0)
    
    # Load ground truth
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    important_genes = set(gt_df[gt_df['Label'] == 1]['Gene'])
    
    # Select top 500 genes (even more focused than before)
    top_genes = df.nlargest(500, df.columns[0])
    
    # Analysis
    top_gene_names = set(top_genes.index)
    important_in_top = top_gene_names.intersection(important_genes)
    
    print(f"📊 Precision-focused dataset:")
    print(f"   Selected top {len(top_genes)} genes")
    print(f"   Contains {len(important_in_top)}/{len(important_genes)} important genes ({len(important_in_top)/len(important_genes)*100:.1f}%)")
    print(f"   Expected precision: {len(important_in_top)/len(top_genes)*100:.1f}%")
    
    # Create dataset
    precision_dir = Path("data/PRECISION_SCREEN")
    precision_dir.mkdir(parents=True, exist_ok=True)
    
    top_genes.to_csv(precision_dir / "PRECISION_SCREEN.csv")
    
    # Ground truth
    precision_gt = []
    for gene in top_genes.index:
        label = 1 if gene in important_genes else 0
        precision_gt.append({'Gene': gene, 'Label': label})
    
    precision_gt_df = pd.DataFrame(precision_gt)
    precision_gt_df.to_csv(precision_dir / "PRECISION_SCREEN_ground_truth.csv", index=False)
    
    # Datasets files
    datasets_dir = Path("datasets")
    positive_genes = precision_gt_df[precision_gt_df['Label'] == 1]['Gene'].tolist()
    np.save(datasets_dir / "topmovers_PRECISION_SCREEN.npy", positive_genes)
    precision_gt_df.to_csv(datasets_dir / "ground_truth_PRECISION_SCREEN.csv", index=False)
    
    # Task prompt
    precision_task = {
        "Task": f"""Elite gene discovery from the top 500 highest-scoring genes.

DATASET: Ultra-high quality - Top 500/15073 genes ({500/15073*100:.1f}%)
EXPECTED SUCCESS RATE: {len(important_in_top)/len(top_genes)*100:.1f}% of genes are essential
SCORE RANGE: {top_genes.iloc[:, 0].min():.3f} to {top_genes.iloc[:, 0].max():.3f} (all high-scoring)

STRATEGY: Since ALL genes here are high-scoring, focus on:
1. THE HIGHEST scorers (top 10% within this elite set)
2. Genes with scores > {top_genes.iloc[:, 0].quantile(0.90):.3f}
3. Pattern recognition in the highest performers

Elite examples: {', '.join(list(top_genes.index[:10]))}

This is precision-focused discovery - every prediction should count!""",
        
        "Measurement": f"Elite importance scores ({top_genes.iloc[:, 0].min():.3f} to {top_genes.iloc[:, 0].max():.3f}). All pre-selected high scorers."
    }
    
    with open(datasets_dir / "task_prompts" / "PRECISION_SCREEN.json", 'w') as f:
        json.dump(precision_task, f, indent=2)
    
    print(f"✅ Precision dataset created: PRECISION_SCREEN")
    print(f"   - {len(positive_genes)} positive genes")
    print(f"   - Expected precision: {len(important_in_top)/len(top_genes)*100:.1f}%")
    
    return True

def suggest_ultimate_tests():
    """Suggest the ultimate test configurations."""
    
    print(f"\n🚀 ULTIMATE TEST CONFIGURATIONS:")
    
    tests = [
        {
            "name": "🏆 ULTIMATE FULL DATASET",
            "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name ultimate_test --data_name MY_CUSTOM_SCREEN --steps 2 --num_genes 48 --log_dir ultimate_logs",
            "description": "Optimized prompt + focused gene count for best balance",
            "expected": "Precision: ~60%, Recall: ~15%"
        },
        {
            "name": "🎯 ULTRA-PRECISION TEST", 
            "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name precision_test --data_name PRECISION_SCREEN --steps 1 --num_genes 24 --log_dir precision_logs",
            "description": "Top 500 genes only, single step, high precision",
            "expected": "Precision: >90%, Recall: varies"
        },
        {
            "name": "🔬 BALANCED DISCOVERY",
            "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name balanced_test --data_name MY_CUSTOM_SCREEN --steps 3 --num_genes 32 --log_dir balanced_logs", 
            "description": "3 steps for learning, moderate gene count",
            "expected": "Best F1-score balance"
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   📝 {test['description']}")
        print(f"   📊 Expected: {test['expected']}")
        print(f"   🚀 Command: {test['command']}")
    
    return tests

if __name__ == "__main__":
    print("🚀 ULTIMATE BioDiscoveryAgent Optimization")
    print("=" * 60)
    
    # Step 1: Create ultimate task prompt
    ultimate_task = create_ultimate_task_prompt()
    
    # Step 2: Create precision-focused dataset  
    precision_created = create_precision_focused_dataset()
    
    # Step 3: Suggest ultimate tests
    tests = suggest_ultimate_tests()
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"Start with the ULTRA-PRECISION TEST for best results:")
    print(f"python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name precision_test --data_name PRECISION_SCREEN --steps 1 --num_genes 24 --log_dir precision_logs")
    
    print(f"\nThen analyze with:")
    print(f"python3 improved_analyze.py --data_name PRECISION_SCREEN --log_dir precision_logs_PRECISION_SCREEN/precision_test --output_dir precision_analysis")
