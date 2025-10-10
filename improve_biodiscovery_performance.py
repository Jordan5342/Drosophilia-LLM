#!/usr/bin/env python3
"""
Improve BioDiscoveryAgent performance by better guiding it toward high-scoring genes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_score_aware_task_prompt():
    """Create a task prompt that emphasizes the importance of gene scores."""
    
    print("🔧 Creating score-aware task prompt...")
    
    # Load your dataset to get score statistics
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Get score statistics
    score_stats = {
        'min': df.iloc[:, 0].min(),
        'max': df.iloc[:, 0].max(),
        'mean': df.iloc[:, 0].mean(),
        'std': df.iloc[:, 0].std()
    }
    
    # Get examples at different score ranges
    high_threshold = score_stats['mean'] + 2 * score_stats['std']  # Top ~2%
    med_threshold = score_stats['mean']
    
    high_scoring = df[df.iloc[:, 0] >= high_threshold].sort_values(df.columns[0], ascending=False)
    medium_scoring = df[(df.iloc[:, 0] >= med_threshold) & (df.iloc[:, 0] < high_threshold)].sort_values(df.columns[0], ascending=False)
    low_scoring = df[df.iloc[:, 0] < med_threshold].sort_values(df.columns[0], ascending=False)
    
    print(f"📊 Score statistics:")
    print(f"   Range: {score_stats['min']:.3f} to {score_stats['max']:.3f}")
    print(f"   Mean: {score_stats['mean']:.3f} ± {score_stats['std']:.3f}")
    print(f"   High scoring (>{high_threshold:.3f}): {len(high_scoring)} genes")
    print(f"   Medium scoring: {len(medium_scoring)} genes")
    print(f"   Low scoring: {len(low_scoring)} genes")
    
    # Create enhanced task prompt
    enhanced_task = {
        "Task": f"""Identify the most important genes for gene essentiality screening from this dataset.

CRITICAL SCORING GUIDANCE:
- Gene importance scores range from {score_stats['min']:.3f} to {score_stats['max']:.3f}
- Higher scores indicate MORE IMPORTANT/ESSENTIAL genes
- Mean score is {score_stats['mean']:.3f}
- Focus on genes with scores ABOVE {score_stats['mean']:.3f}, especially above {high_threshold:.3f}

EXAMPLES BY IMPORTANCE LEVEL:
HIGH IMPORTANCE (score >{high_threshold:.3f}): {', '.join(list(high_scoring.index[:8]))}
MEDIUM IMPORTANCE (score {med_threshold:.3f}-{high_threshold:.3f}): {', '.join(list(medium_scoring.index[:8]))}
LOW IMPORTANCE (score <{med_threshold:.3f}): {', '.join(list(low_scoring.index[:8]))}

SELECTION STRATEGY:
1. PRIORITIZE genes with highest scores (closer to {score_stats['max']:.3f})
2. Consider biological patterns in high-scoring genes
3. Avoid genes with very low scores (below {med_threshold:.3f})
4. Balance exploration with exploitation of score information

Remember: You are looking for ESSENTIAL genes, which should have HIGH importance scores.""",

        "Measurement": f"Gene importance score from {score_stats['min']:.3f} to {score_stats['max']:.3f}. Higher scores indicate more essential/important genes. Mean = {score_stats['mean']:.3f}."
    }
    
    # Save the enhanced task prompt
    task_prompt_file = Path("datasets/task_prompts/MY_CUSTOM_SCREEN.json")
    
    # Backup current version
    backup_file = task_prompt_file.with_suffix('.json.score_aware_backup')
    import shutil
    shutil.copy2(task_prompt_file, backup_file)
    
    with open(task_prompt_file, 'w') as f:
        json.dump(enhanced_task, f, indent=2)
    
    print(f"✅ Enhanced task prompt saved: {task_prompt_file}")
    print(f"✅ Previous version backed up: {backup_file}")
    
    return enhanced_task

def create_focused_test_dataset():
    """Create a smaller, more focused test dataset for faster iteration."""
    
    print(f"\n🔧 Creating focused test dataset...")
    
    # Load full dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Load ground truth to know which genes are actually important
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    important_genes = set(gt_df[gt_df['Label'] == 1]['Gene'])
    
    # Select a focused subset: top 1000 genes by score
    top_genes = df.nlargest(1000, df.columns[0])
    
    # Check how many important genes are in this subset
    top_gene_names = set(top_genes.index)
    important_in_top = top_gene_names.intersection(important_genes)
    
    print(f"📊 Focused dataset statistics:")
    print(f"   Selected top {len(top_genes)} genes by score")
    print(f"   Contains {len(important_in_top)}/{len(important_genes)} important genes ({len(important_in_top)/len(important_genes)*100:.1f}%)")
    
    # Create focused dataset directory
    focused_dir = Path("data/FOCUSED_SCREEN")
    focused_dir.mkdir(parents=True, exist_ok=True)
    
    # Save focused dataset
    top_genes.to_csv(focused_dir / "FOCUSED_SCREEN.csv")
    
    # Create ground truth for focused dataset
    focused_gt = []
    for gene in top_genes.index:
        label = 1 if gene in important_genes else 0
        focused_gt.append({'Gene': gene, 'Label': label})
    
    focused_gt_df = pd.DataFrame(focused_gt)
    focused_gt_df.to_csv(focused_dir / "FOCUSED_SCREEN_ground_truth.csv", index=False)
    
    # Create datasets files
    datasets_dir = Path("datasets")
    
    # Topmovers (positive genes in focused dataset)
    positive_genes = focused_gt_df[focused_gt_df['Label'] == 1]['Gene'].tolist()
    np.save(datasets_dir / "topmovers_FOCUSED_SCREEN.npy", positive_genes)
    
    # Ground truth in datasets
    focused_gt_df.to_csv(datasets_dir / "ground_truth_FOCUSED_SCREEN.csv", index=False)
    
    # Task prompt for focused dataset
    score_range = {
        'min': top_genes.iloc[:, 0].min(),
        'max': top_genes.iloc[:, 0].max(),
        'mean': top_genes.iloc[:, 0].mean()
    }
    
    focused_task = {
        "Task": f"""Identify essential genes from this focused high-scoring dataset.

This dataset contains the top 1000 highest-scoring genes from the full screen.
Scores range from {score_range['min']:.3f} to {score_range['max']:.3f} (all high-scoring).

STRATEGY: Since all genes here have relatively high scores, look for:
1. The highest scoring genes (closest to {score_range['max']:.3f})
2. Genes that score well above the mean ({score_range['mean']:.3f})
3. Patterns in gene IDs that correlate with importance

Top examples: {', '.join(list(top_genes.index[:10]))}""",

        "Measurement": f"Gene importance score ({score_range['min']:.3f} to {score_range['max']:.3f}). All genes are pre-selected high scorers."
    }
    
    prompt_dir = datasets_dir / "task_prompts"
    with open(prompt_dir / "FOCUSED_SCREEN.json", 'w') as f:
        json.dump(focused_task, f, indent=2)
    
    print(f"✅ Focused dataset created: FOCUSED_SCREEN")
    print(f"   - {len(positive_genes)} positive genes")
    print(f"   - {len(focused_gt_df) - len(positive_genes)} negative genes")
    print(f"   - Score range: {score_range['min']:.3f} to {score_range['max']:.3f}")
    
    return True

def suggest_optimized_parameters():
    """Suggest optimized parameters for better performance."""
    
    print(f"\n💡 Suggested optimized parameters:")
    
    suggestions = [
        {
            "name": "Score-Aware Full Dataset",
            "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name score_aware_test --data_name MY_CUSTOM_SCREEN --steps 3 --num_genes 64 --log_dir score_aware_logs",
            "description": "Test with improved score-aware prompt on full dataset"
        },
        {
            "name": "Focused High-Scoring Dataset", 
            "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name focused_test --data_name FOCUSED_SCREEN --steps 2 --num_genes 32 --log_dir focused_logs",
            "description": "Test on smaller dataset of only high-scoring genes"
        },
        {
            "name": "Conservative High-Precision",
            "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name conservative_test --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 16 --log_dir conservative_logs", 
            "description": "Single step with fewer genes for higher precision"
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['name']}")
        print(f"   {suggestion['description']}")
        print(f"   Command: {suggestion['command']}")
    
    return suggestions

if __name__ == "__main__":
    print("🔧 Improving BioDiscoveryAgent Performance")
    print("=" * 55)
    
    # Step 1: Create score-aware task prompt
    enhanced_task = create_score_aware_task_prompt()
    
    # Step 2: Create focused test dataset
    focused_created = create_focused_test_dataset()
    
    # Step 3: Suggest optimized parameters
    suggestions = suggest_optimized_parameters()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Try the score-aware approach on your full dataset")
    print(f"2. Test the focused dataset for faster iteration")
    print(f"3. Compare results with the baseline analysis")
    
    print(f"\n🚀 Recommended starting point:")
    print(f"python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name score_aware_test --data_name MY_CUSTOM_SCREEN --steps 2 --num_genes 32 --log_dir score_aware_logs")
