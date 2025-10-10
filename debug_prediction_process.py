#!/usr/bin/env python3
"""
Strengthen the gene format guidance to ensure consistent predictions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_enhanced_task_prompt():
    """Create a much more explicit task prompt with gene examples."""
    
    print("🔧 Creating enhanced task prompt with explicit gene examples...")
    
    # Read your dataset to get actual genes
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Get high, medium, and low scoring genes as examples
    top_genes = df.nlargest(50, df.columns[0])
    bottom_genes = df.nsmallest(50, df.columns[0])
    middle_genes = df.iloc[len(df)//3:len(df)//3+50]  # Middle range
    
    # Create comprehensive examples
    high_examples = list(top_genes.index[:10])
    med_examples = list(middle_genes.index[:10])
    low_examples = list(bottom_genes.index[:10])
    
    print(f"📊 Gene examples from your dataset:")
    print(f"   High scoring: {high_examples[:5]}")
    print(f"   Medium scoring: {med_examples[:5]}")
    print(f"   Low scoring: {low_examples[:5]}")
    
    # Create a very explicit task prompt
    enhanced_task = {
        "Task": f"""Identify genes important for gene_essentiality_screen from the provided dataset.

CRITICAL INSTRUCTIONS FOR GENE SELECTION:
- You MUST select genes that exist in the provided dataset
- ALL genes in this dataset use the format GENE_XXXXX where XXXXX is a number
- Do NOT use biological gene names like CDK2, TP53, BRCA1, MYC, etc.
- ONLY use genes from the dataset in format GENE_XXXXX

EXAMPLES OF VALID GENES IN THIS DATASET:
High importance: {', '.join(high_examples[:5])}
Medium importance: {', '.join(med_examples[:5])}
Low importance: {', '.join(low_examples[:5])}

When making predictions, you must:
1. Consider the gene importance scores provided
2. Select genes in the GENE_XXXXX format only
3. Choose genes that exist in the dataset
4. Focus on genes with high scores for positive selection

Remember: This is a numbered gene dataset, not a biological gene name dataset.""",

        "Measurement": "Gene importance score derived from construct frequency and biological relevance. Scores range from 0.01 to 1.0, where higher scores indicate more important/essential genes."
    }
    
    # Save the enhanced task prompt
    task_prompt_file = Path("datasets/task_prompts/MY_CUSTOM_SCREEN.json")
    
    # Create additional backup
    backup_file = task_prompt_file.with_suffix('.json.original')
    if not backup_file.exists():
        import shutil
        shutil.copy2(task_prompt_file, backup_file)
    
    with open(task_prompt_file, 'w') as f:
        json.dump(enhanced_task, f, indent=2)
    
    print(f"✅ Created enhanced task prompt: {task_prompt_file}")
    print(f"✅ Original backup: {backup_file}")
    
    return enhanced_task

def create_gene_reference_file():
    """Create a reference file with available genes for the AI to use."""
    
    print(f"\n🔧 Creating gene reference file...")
    
    # Read your dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Sort genes by score (highest first)
    df_sorted = df.sort_values(df.columns[0], ascending=False)
    
    # Create a reference file with gene information
    gene_reference = {
        "dataset_info": {
            "total_genes": len(df),
            "score_range": {
                "min": float(df[df.columns[0]].min()),
                "max": float(df[df.columns[0]].max()),
                "mean": float(df[df.columns[0]].mean())
            },
            "gene_format": "GENE_XXXXX where XXXXX is a number",
            "example_genes": list(df.index[:20])
        },
        "high_scoring_genes": {
            "description": "Top 100 highest scoring genes (most important)",
            "genes": list(df_sorted.index[:100])
        },
        "medium_scoring_genes": {
            "description": "Medium scoring genes",
            "genes": list(df_sorted.index[len(df)//3:len(df)//3+100])
        },
        "low_scoring_genes": {
            "description": "Bottom 100 lowest scoring genes (least important)",
            "genes": list(df_sorted.index[-100:])
        }
    }
    
    # Save reference file
    ref_file = Path("data/MY_CUSTOM_SCREEN/gene_reference.json")
    with open(ref_file, 'w') as f:
        json.dump(gene_reference, f, indent=2)
    
    print(f"✅ Created gene reference: {ref_file}")
    
    return gene_reference

def test_with_minimal_genes():
    """Create a very small test dataset with obvious patterns."""
    
    print(f"\n🔧 Creating minimal test dataset with obvious patterns...")
    
    # Create a tiny dataset with clear patterns
    test_genes = [f"GENE_{i}" for i in range(1, 51)]  # 50 genes total
    
    # Create very obvious score patterns
    scores = []
    for i, gene in enumerate(test_genes):
        if i < 10:  # First 10 genes have high scores
            scores.append(0.9 + i * 0.01)  # 0.90-0.99
        elif i < 20:  # Next 10 have medium scores  
            scores.append(0.5 + (i-10) * 0.01)  # 0.50-0.59
        else:  # Rest have low scores
            scores.append(0.1 + (i-20) * 0.001)  # 0.10-0.13
    
    # Create test dataset
    test_df = pd.DataFrame({
        'Score': scores
    }, index=test_genes)
    
    # Create test directory
    test_dir = Path("data/TEST_SIMPLE")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    test_df.to_csv(test_dir / "TEST_SIMPLE.csv")
    
    # Ground truth: top 5 as positive, bottom 5 as negative
    gt_df = pd.DataFrame({
        'Gene': test_genes[:5] + test_genes[-5:],
        'Label': [1] * 5 + [0] * 5
    })
    gt_df.to_csv(test_dir / "TEST_SIMPLE_ground_truth.csv", index=False)
    
    # Create datasets files
    datasets_dir = Path("datasets")
    
    # Topmovers (top 5)
    np.save(datasets_dir / "topmovers_TEST_SIMPLE.npy", test_genes[:5])
    
    # Ground truth
    gt_df.to_csv(datasets_dir / "ground_truth_TEST_SIMPLE.csv", index=False)
    
    # Task prompt with explicit gene examples
    task_prompt = {
        "Task": f"""Identify important genes from this test dataset.

CRITICAL: This dataset contains exactly these genes: {', '.join(test_genes)}

You MUST select genes from this exact list. Do NOT use biological gene names.

The highest scoring genes are: {', '.join(test_genes[:5])}
The lowest scoring genes are: {', '.join(test_genes[-5:])}

Select genes that exist in this dataset only.""",
        "Measurement": "Gene importance score (0.1 to 1.0). Higher = more important."
    }
    
    prompt_dir = datasets_dir / "task_prompts"
    prompt_dir.mkdir(exist_ok=True)
    with open(prompt_dir / "TEST_SIMPLE.json", 'w') as f:
        json.dump(task_prompt, f, indent=2)
    
    print(f"✅ Created simple test dataset: TEST_SIMPLE")
    print(f"   - 50 genes with clear score patterns")
    print(f"   - Top 5 genes: {test_genes[:5]}")
    print(f"   - Bottom 5 genes: {test_genes[-5:]}")
    
    return True

if __name__ == "__main__":
    print("🔧 Strengthening Gene Format Guidance")
    print("=" * 50)
    
    # Step 1: Create enhanced task prompt
    enhanced_task = create_enhanced_task_prompt()
    
    # Step 2: Create gene reference file
    gene_ref = create_gene_reference_file()
    
    # Step 3: Create simple test dataset
    test_created = test_with_minimal_genes()
    
    print(f"\n🚀 Try these commands:")
    
    print(f"\n1. Test with enhanced prompt on your dataset:")
    print(f"python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name enhanced_test --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 8 --log_dir enhanced_logs")
    
    print(f"\n2. Test with simple dataset (50 genes, clear patterns):")  
    print(f"python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name simple_test --data_name TEST_SIMPLE --steps 1 --num_genes 8 --log_dir simple_logs")
    
    print(f"\n💡 The enhanced prompt now:")
    print(f"   - Explicitly lists gene examples from your dataset")
    print(f"   - Warns against biological gene names")
    print(f"   - Provides scoring context")
    print(f"   - Uses smaller gene count (8) for easier testing")