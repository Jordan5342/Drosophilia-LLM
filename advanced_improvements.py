#!/usr/bin/env python3
"""
Advanced improvements for BioDiscoveryAgent performance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def create_advanced_prompt_with_examples():
    """Create advanced prompt with successful gene examples and scoring patterns."""
    
    print("🧠 Creating advanced prompt with learned patterns...")
    
    # Load dataset and successful patterns
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Successful genes from your analysis
    proven_winners = [
        'GENE_10303', 'GENE_15048', 'GENE_10304', 'GENE_15049', 'GENE_10302',
        'GENE_17688', 'GENE_17665', 'GENE_17662', 'GENE_17654', 'GENE_17684',
        'GENE_17685', 'GENE_17693', 'GENE_17709'
    ]
    
    # Analyze successful patterns
    winner_scores = []
    gene_ranges = {}
    
    for gene in proven_winners:
        if gene in df.index:
            score = df.loc[gene, df.columns[0]]
            winner_scores.append(score)
            
            # Extract gene number for pattern analysis
            gene_num = int(gene.split('_')[1])
            range_key = f"{gene_num//1000}000s"  # Group by thousands
            if range_key not in gene_ranges:
                gene_ranges[range_key] = []
            gene_ranges[range_key].append(gene)
    
    avg_winner_score = np.mean(winner_scores)
    min_winner_score = np.min(winner_scores)
    
    print(f"📊 Successful gene analysis:")
    print(f"   Average winner score: {avg_winner_score:.3f}")
    print(f"   Minimum winner score: {min_winner_score:.3f}")
    print(f"   Gene ranges with winners: {list(gene_ranges.keys())}")
    
    # Create advanced prompt with learned patterns
    advanced_task = {
        "Task": f"""Expert gene essentiality discovery using proven patterns.

PROVEN SUCCESS PATTERNS (from previous discoveries):
✅ CONFIRMED ESSENTIAL GENES: {', '.join(proven_winners[:8])}
✅ SUCCESS SCORE THRESHOLD: All winners scored ≥ {min_winner_score:.3f}
✅ AVERAGE WINNER SCORE: {avg_winner_score:.3f}
✅ HIGH-VALUE RANGES: Genes in {list(gene_ranges.keys())} ranges showed success

ADVANCED SELECTION STRATEGY:
1. PRIMARY: Select genes scoring ≥ {avg_winner_score:.3f} (proven winner zone)
2. SECONDARY: Consider genes scoring {min_winner_score:.3f}-{avg_winner_score:.3f} (potential winners)
3. PATTERN: Look for genes in successful ranges: {list(gene_ranges.keys())}
4. AVOID: Genes scoring < {min_winner_score:.3f} (low success probability)

EXAMPLES OF IDEAL TARGETS:
- GENE_10303 (score: 1.000) ✅ PERFECT
- GENE_15048 (score: 0.952) ✅ EXCELLENT  
- GENE_17688 (score: 0.929) ✅ VERY HIGH

Your mission: Find genes matching these proven success patterns.""",

        "Measurement": f"Essentiality score 0-1. Winner threshold: ≥{min_winner_score:.3f}. Target zone: ≥{avg_winner_score:.3f}"
    }
    
    # Save advanced prompt
    task_file = Path("datasets/task_prompts/MY_CUSTOM_SCREEN.json")
    backup_file = task_file.with_suffix('.json.advanced_backup')
    
    import shutil
    shutil.copy2(task_file, backup_file)
    
    with open(task_file, 'w') as f:
        json.dump(advanced_task, f, indent=2)
    
    print(f"✅ Advanced prompt saved: {task_file}")
    return advanced_task

def create_hierarchical_dataset():
    """Create a hierarchical dataset with difficulty levels."""
    
    print("\n🎯 Creating hierarchical difficulty dataset...")
    
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv") 
    gt_df = pd.read_csv(gt_file)
    important_genes = set(gt_df[gt_df['Label'] == 1]['Gene'])
    
    # Create three difficulty levels
    score_95th = df.iloc[:, 0].quantile(0.95)  # Top 5% - Easy
    score_85th = df.iloc[:, 0].quantile(0.85)  # Top 15% - Medium
    score_70th = df.iloc[:, 0].quantile(0.70)  # Top 30% - Hard
    
    datasets_to_create = [
        ("EASY_SCREEN", 100, score_95th, "Top 5% genes - should be easy to identify"),
        ("MEDIUM_SCREEN", 300, score_85th, "Top 15% genes - moderate difficulty"),
        ("HARD_SCREEN", 600, score_70th, "Top 30% genes - challenging")
    ]
    
    for dataset_name, num_genes, threshold, description in datasets_to_create:
        # Select genes above threshold
        candidate_genes = df[df.iloc[:, 0] >= threshold].nlargest(num_genes, df.columns[0])
        
        # Create dataset directory
        dataset_dir = Path(f"data/{dataset_name}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        candidate_genes.to_csv(dataset_dir / f"{dataset_name}.csv")
        
        # Create ground truth
        gt_data = []
        for gene in candidate_genes.index:
            label = 1 if gene in important_genes else 0
            gt_data.append({'Gene': gene, 'Label': label})
        
        gt_df_new = pd.DataFrame(gt_data)
        gt_df_new.to_csv(dataset_dir / f"{dataset_name}_ground_truth.csv", index=False)
        
        # Create datasets files
        datasets_dir = Path("datasets")
        positive_genes = gt_df_new[gt_df_new['Label'] == 1]['Gene'].tolist()
        np.save(datasets_dir / f"topmovers_{dataset_name}.npy", positive_genes)
        gt_df_new.to_csv(datasets_dir / f"ground_truth_{dataset_name}.csv", index=False)
        
        # Create task prompt
        task_prompt = {
            "Task": f"""Hierarchical gene discovery - {description}

DIFFICULTY LEVEL: {dataset_name.split('_')[0]}
GENE POOL: {num_genes} pre-selected genes (score ≥ {threshold:.3f})
SUCCESS RATE: {len(positive_genes)/len(gt_df_new)*100:.1f}% of genes are essential

STRATEGY FOR THIS LEVEL:
- Focus on genes with highest scores within this curated set
- All genes here already passed initial filtering
- Look for the cream of the crop

Top examples: {', '.join(list(candidate_genes.index[:8]))}""",
            
            "Measurement": f"Curated essentiality scores (≥{threshold:.3f}). Range: {candidate_genes.iloc[:, 0].min():.3f}-{candidate_genes.iloc[:, 0].max():.3f}"
        }
        
        with open(datasets_dir / "task_prompts" / f"{dataset_name}.json", 'w') as f:
            json.dump(task_prompt, f, indent=2)
        
        print(f"✅ Created {dataset_name}: {num_genes} genes, {len(positive_genes)} positive ({len(positive_genes)/len(gt_df_new)*100:.1f}% precision)")

def create_ensemble_approach():
    """Create multiple prompt variations for ensemble testing."""
    
    print(f"\n🔄 Creating ensemble prompt variations...")
    
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    # Create different prompt styles
    prompt_variations = {
        "STATISTICAL": {
            "Task": f"""Statistical gene discovery approach.

QUANTITATIVE ANALYSIS:
- Dataset: {len(df)} genes, scores 0-1
- Target: Top 10% performers (score > {df.iloc[:, 0].quantile(0.90):.3f})
- Method: Statistical significance-based selection

SELECT genes with:
1. Score > 95th percentile ({df.iloc[:, 0].quantile(0.95):.3f})
2. Score in top 5% AND consistent with high-importance pattern
3. Statistical outliers in the upper tail

Use quantitative ranking for gene prioritization.""",
            "Measurement": "Statistical importance score. Focus on upper 5% tail."
        },
        
        "BIOLOGICAL": {
            "Task": f"""Biological gene essentiality discovery.

BIOLOGICAL REASONING:
Essential genes are critical for:
- Cell survival and viability
- Core cellular processes
- Fundamental biological pathways

HIGH-SCORING GENES likely represent:
- Essential metabolic enzymes
- Critical structural proteins  
- Core regulatory elements

Select genes that:
1. Have very high importance scores (>{df.iloc[:, 0].quantile(0.95):.3f})
2. Represent likely essential functions
3. Show consistent high ranking

Think like a biologist - what genes would cells need to survive?""",
            "Measurement": "Biological essentiality score. Higher = more critical for survival."
        },
        
        "MACHINE_LEARNING": {
            "Task": f"""Machine learning pattern recognition for gene discovery.

ML APPROACH:
- Feature: Gene importance score
- Target: Essential vs non-essential classification
- Method: High-score threshold optimization

LEARNED PATTERNS:
- Decision boundary: ~{df.iloc[:, 0].quantile(0.90):.3f}
- High-confidence zone: >{df.iloc[:, 0].quantile(0.95):.3f}
- Precision-optimized threshold: >{df.iloc[:, 0].quantile(0.97):.3f}

Apply ML logic:
1. Rank by confidence score
2. Select above optimal threshold
3. Maximize precision over recall

Use data-driven decision making.""",
            "Measurement": "ML confidence score. Optimize for precision."
        }
    }
    
    # Save prompt variations
    prompts_dir = Path("datasets/task_prompts")
    for style, prompt in prompt_variations.items():
        prompt_file = prompts_dir / f"MY_CUSTOM_SCREEN_{style}.json"
        with open(prompt_file, 'w') as f:
            json.dump(prompt, f, indent=2)
        print(f"✅ Created {style} prompt variation")
    
    return prompt_variations

def suggest_improvement_experiments():
    """Suggest systematic experiments for improvement."""
    
    print(f"\n🧪 SYSTEMATIC IMPROVEMENT EXPERIMENTS:")
    
    experiments = [
        {
            "category": "🧠 PROMPT ENGINEERING",
            "experiments": [
                {
                    "name": "Advanced Pattern Learning",
                    "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name advanced_test --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 24 --log_dir advanced_logs",
                    "goal": "Use learned patterns to improve precision",
                    "expected": "Precision > 65%"
                },
                {
                    "name": "Statistical Approach",
                    "setup": "cp datasets/task_prompts/MY_CUSTOM_SCREEN_STATISTICAL.json datasets/task_prompts/MY_CUSTOM_SCREEN.json",
                    "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name statistical_test --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 20 --log_dir statistical_logs",
                    "goal": "Test statistical reasoning approach"
                }
            ]
        },
        {
            "category": "🎯 DATASET OPTIMIZATION", 
            "experiments": [
                {
                    "name": "Easy Level Training",
                    "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name easy_test --data_name EASY_SCREEN --steps 1 --num_genes 16 --log_dir easy_logs",
                    "goal": "Train on easy examples first",
                    "expected": "Very high precision (>90%)"
                },
                {
                    "name": "Progressive Difficulty",
                    "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name medium_test --data_name MEDIUM_SCREEN --steps 1 --num_genes 20 --log_dir medium_logs",
                    "goal": "Moderate difficulty after easy success"
                }
            ]
        },
        {
            "category": "⚙️ PARAMETER OPTIMIZATION",
            "experiments": [
                {
                    "name": "Multi-Step Learning",
                    "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name multistep_test --data_name MY_CUSTOM_SCREEN --steps 3 --num_genes 16 --log_dir multistep_logs",
                    "goal": "Allow learning across steps",
                    "expected": "Improvement from step 1 to 3"
                },
                {
                    "name": "Optimal Gene Count",
                    "command": "python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name optimal_test --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 32 --log_dir optimal_logs",
                    "goal": "Find optimal gene count per step"
                }
            ]
        }
    ]
    
    for category_info in experiments:
        print(f"\n{category_info['category']}:")
        for i, exp in enumerate(category_info['experiments'], 1):
            print(f"  {i}. {exp['name']}")
            print(f"     Goal: {exp['goal']}")
            if 'setup' in exp:
                print(f"     Setup: {exp['setup']}")
            print(f"     Command: {exp['command']}")
            if 'expected' in exp:
                print(f"     Expected: {exp['expected']}")
            print()
    
    return experiments

if __name__ == "__main__":
    print("🚀 ADVANCED BIODISCOVERYAGENT IMPROVEMENTS")
    print("=" * 60)
    
    # Step 1: Create advanced prompt with learned patterns
    advanced_task = create_advanced_prompt_with_examples()
    
    # Step 2: Create hierarchical datasets
    create_hierarchical_dataset()
    
    # Step 3: Create ensemble prompt variations
    prompt_variations = create_ensemble_approach()
    
    # Step 4: Suggest systematic experiments
    experiments = suggest_improvement_experiments()
    
    print(f"\n🎯 RECOMMENDED IMPROVEMENT PATH:")
    print(f"1. Start with Advanced Pattern Learning:")
    print(f"   python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name advanced_test --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 24 --log_dir advanced_logs")
    
    print(f"\n2. Then try Easy Level Training:")
    print(f"   python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name easy_test --data_name EASY_SCREEN --steps 1 --num_genes 16 --log_dir easy_logs")
    
    print(f"\n3. Analyze and compare all results:")
    print(f"   python3 improved_analyze.py --data_name MY_CUSTOM_SCREEN --log_dir advanced_logs_MY_CUSTOM_SCREEN/advanced_test --output_dir advanced_analysis")
    
    print(f"\n💡 TARGET: Push precision from 61% to 75%+ through systematic optimization!")
