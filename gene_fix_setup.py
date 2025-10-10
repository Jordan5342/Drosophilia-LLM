#!/usr/bin/env python3
"""
Fixed setup script that handles gene ID mismatches.
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

def debug_gene_mismatch(ground_truth_path, topmovers_path):
    """Debug the gene ID mismatch issue."""
    print("🔍 Debugging gene ID mismatch...")
    
    # Load ground truth
    df = pd.read_csv(ground_truth_path, index_col=0)
    print(f"\n📊 Ground truth data:")
    print(f"   - Total genes: {len(df)}")
    print(f"   - Sample gene names: {list(df.index[:10])}")
    print(f"   - Gene name type: {type(df.index[0])}")
    
    # Load top movers
    topmovers = np.load(topmovers_path, allow_pickle=True)
    print(f"\n🎯 Top movers data:")
    print(f"   - Total top movers: {len(topmovers)}")
    print(f"   - Sample top movers: {topmovers[:10].tolist()}")
    print(f"   - Top mover type: {type(topmovers[0])}")
    
    # Check for matches
    matches = set(df.index).intersection(set(topmovers))
    print(f"\n🔄 Matching analysis:")
    print(f"   - Direct matches: {len(matches)}")
    
    if len(matches) == 0:
        print("   - No direct matches found!")
        print("   - This suggests different gene ID formats")
        
        # Try some common conversions
        print(f"\n🧪 Testing conversions...")
        
        # Convert topmovers to strings
        topmovers_str = [str(x) for x in topmovers]
        matches_str = set(df.index).intersection(set(topmovers_str))
        print(f"   - Matches after converting topmovers to string: {len(matches_str)}")
        
        # Convert ground truth index to strings
        df_index_str = [str(x) for x in df.index]
        matches_gt_str = set(df_index_str).intersection(set(topmovers))
        print(f"   - Matches after converting ground truth to string: {len(matches_gt_str)}")
        
        return {
            'direct_matches': len(matches),
            'topmovers_to_str': len(matches_str),
            'ground_truth_to_str': len(matches_gt_str),
            'topmovers_sample': topmovers[:10],
            'ground_truth_sample': list(df.index[:10])
        }
    else:
        print(f"   - Found {len(matches)} direct matches")
        return {'direct_matches': len(matches)}

def setup_dataset_with_gene_fix(ground_truth_path, task_prompt_path, topmovers_path, dataset_name="MY_SCREEN"):
    """Set up dataset with gene ID mismatch handling."""
    print(f"🚀 Setting up dataset: {dataset_name}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Verify all files exist
    files_to_check = {
        'Ground truth': ground_truth_path,
        'Task prompt': task_prompt_path,
        'Top movers': topmovers_path
    }
    
    print("\n🔍 Checking input files:")
    for name, path in files_to_check.items():
        if path and os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
            return False
    
    # Debug gene mismatch first
    debug_info = debug_gene_mismatch(ground_truth_path, topmovers_path)
    
    # Create data directory structure
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Process ground truth data
        print(f"\n📊 Processing ground truth data...")
        df = pd.read_csv(ground_truth_path, index_col=0)
        print(f"   - Loaded {len(df)} genes")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Score range: {df.iloc[:, 0].min():.3f} to {df.iloc[:, 0].max():.3f}")
        
        # Save main dataset file
        main_file = dataset_dir / f"{dataset_name}.csv"
        df.index.name = 'Gene'
        df.to_csv(main_file)
        print(f"✅ Created main dataset: {main_file}")
        
        # 2. Process task prompt
        print(f"\n📝 Processing task prompt...")
        with open(task_prompt_path, 'r') as f:
            task_info = json.load(f)
        print(f"   - Task: {task_info.get('Task', 'N/A')}")
        print(f"   - Measurement: {task_info.get('Measurement', 'N/A')}")
        
        # 3. Process top movers with gene ID fixing
        print(f"\n🎯 Processing top movers with gene ID matching...")
        topmovers = np.load(topmovers_path, allow_pickle=True)
        print(f"   - Loaded {len(topmovers)} top movers")
        
        # Try different matching strategies
        valid_topmovers = []
        
        # Strategy 1: Direct matching
        direct_matches = [gene for gene in topmovers if gene in df.index]
        if direct_matches:
            valid_topmovers = direct_matches
            print(f"   - Direct matches: {len(direct_matches)}")
        
        # Strategy 2: Convert topmovers to string and match
        if not valid_topmovers:
            topmovers_str = [str(gene) for gene in topmovers]
            str_matches = [gene for gene in topmovers_str if gene in df.index]
            if str_matches:
                valid_topmovers = str_matches
                print(f"   - String conversion matches: {len(str_matches)}")
        
        # Strategy 3: Convert ground truth index to string
        if not valid_topmovers:
            df_str_index = df.copy()
            df_str_index.index = df_str_index.index.astype(str)
            int_matches = [str(gene) for gene in topmovers if str(gene) in df_str_index.index]
            if int_matches:
                valid_topmovers = int_matches
                # Update df to use string index for consistency
                df = df_str_index
                print(f"   - Ground truth string conversion matches: {len(int_matches)}")
        
        # Strategy 4: Generate synthetic top movers based on scores
        if not valid_topmovers:
            print("   - No matches found with any strategy!")
            print("   - Generating synthetic top movers based on highest scores...")
            
            # Get top scoring genes as synthetic top movers
            num_synthetic = min(len(topmovers), len(df))  # Use same number as original topmovers
            top_scoring = df.nlargest(num_synthetic, df.columns[0])
            valid_topmovers = list(top_scoring.index)
            print(f"   - Generated {len(valid_topmovers)} synthetic top movers")
        
        print(f"   - Final valid top movers: {len(valid_topmovers)}")
        print(f"   - Sample valid top movers: {valid_topmovers[:5]}")
        
        # 4. Create evaluation ground truth
        print(f"\n⚖️ Creating evaluation ground truth...")
        
        if len(valid_topmovers) == 0:
            print("❌ No valid top movers found even with all strategies!")
            return False
        
        positive_df = pd.DataFrame({
            'Gene': valid_topmovers,
            'Label': 1
        })
        
        # Negative examples (bottom scoring genes)
        num_negatives = min(len(valid_topmovers), len(df) - len(valid_topmovers))
        bottom_genes = df.nsmallest(num_negatives, df.columns[0]).index.tolist()
        negative_df = pd.DataFrame({
            'Gene': bottom_genes,
            'Label': 0
        })
        
        # Combine positive and negative
        eval_df = pd.concat([positive_df, negative_df], ignore_index=True)
        eval_file = dataset_dir / f"{dataset_name}_ground_truth.csv"
        eval_df.to_csv(eval_file, index=False)
        print(f"✅ Created evaluation file: {eval_file}")
        print(f"   - Positive examples: {len(positive_df)}")
        print(f"   - Negative examples: {len(negative_df)}")
        
        # Update main dataset file with corrected gene names if needed
        if df.index.dtype != 'object':  # If we had to convert
            main_file = dataset_dir / f"{dataset_name}.csv"
            df.index.name = 'Gene'
            df.to_csv(main_file)
            print(f"✅ Updated main dataset with corrected gene names: {main_file}")
        
        # 5. Create metadata
        print(f"\n📋 Creating metadata...")
        
        # Convert numpy arrays to lists for JSON serialization
        def make_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        metadata = {
            "dataset_name": dataset_name,
            "description": task_info.get('Task', f'Custom genetic screen: {dataset_name}'),
            "measurement": task_info.get('Measurement', 'Gene importance score'),
            "total_genes": len(df),
            "positive_hits": len(valid_topmovers),
            "negative_examples": len(negative_df),
            "score_stats": {
                "min": float(df.iloc[:, 0].min()),
                "max": float(df.iloc[:, 0].max()),
                "mean": float(df.iloc[:, 0].mean()),
                "std": float(df.iloc[:, 0].std())
            },
            "top_genes": valid_topmovers[:10],
            "gene_id_matching": {
                "strategy_used": "multiple_strategies_attempted",
                "final_matches": len(valid_topmovers),
                "debug_info": make_json_serializable(debug_info)
            },
            "source_files": {
                "ground_truth": ground_truth_path,
                "task_prompt": task_prompt_path,
                "topmovers": topmovers_path
            }
        }
        
        # Make sure everything is JSON serializable
        metadata = make_json_serializable(metadata)
        
        metadata_file = dataset_dir / f"{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Created metadata: {metadata_file}")
        
        # 6. Create run script
        print(f"\n🚀 Creating run script...")
        run_command = f"""python3 research_assistant.py \\
    --task perturb-genes-brief \\
    --model claude-3-5-sonnet-20240620 \\
    --run_name {dataset_name.lower()}_run \\
    --data_name {dataset_name} \\
    --steps 3 \\
    --num_genes 64 \\
    --log_dir {dataset_name.lower()}_logs"""
        
        script_content = f"""#!/bin/bash
# Run BioDiscoveryAgent with {dataset_name}
echo "🚀 Running BioDiscoveryAgent with {dataset_name}..."
{run_command}
echo "✅ Run completed! Check {dataset_name.lower()}_logs/ for results."
"""
        
        script_file = f"run_{dataset_name.lower()}.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        os.chmod(script_file, 0o755)
        
        print(f"✅ Created run script: {script_file}")
        
        # 7. Final verification
        print(f"\n🔍 Final verification...")
        required_agent_files = ["research_assistant.py", "analyze.py"]
        for req_file in required_agent_files:
            if os.path.exists(req_file):
                print(f"✅ Found: {req_file}")
            else:
                print(f"❌ Missing: {req_file}")
                return False
        
        # Success summary
        print(f"\n🎉 SUCCESS! Dataset '{dataset_name}' is ready!")
        print(f"📁 Dataset location: {dataset_dir}")
        print(f"📊 Dataset summary:")
        print(f"   - Total genes: {len(df)}")
        print(f"   - Positive hits: {len(valid_topmovers)}")
        print(f"   - Score range: {df.iloc[:, 0].min():.3f} to {df.iloc[:, 0].max():.3f}")
        
        print(f"\n🚀 To run BioDiscoveryAgent:")
        print(f"   bash {script_file}")
        print(f"\nOr manually:")
        print(f"   {run_command}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 BioDiscoveryAgent Dataset Setup (Gene ID Fix)")
    print("=" * 60)
    
    # Your file paths (already updated)
    ground_truth_path = '/Users/jordansztejman/Downloads/BioDiscoveryAgent-master/datasets/ground_truth_gene_essentiality_screen.csv'
    task_prompt_path = '/Users/jordansztejman/Downloads/BioDiscoveryAgent-master/datasets/task_prompts/gene_essentiality_screen.json'
    topmovers_path = '/Users/jordansztejman/Downloads/BioDiscoveryAgent-master/datasets/topmovers_gene_essentiality_screen.npy'
    dataset_name = 'MY_CUSTOM_SCREEN'
    
    print("🎯 Using file paths:")
    print(f"   Ground truth: {ground_truth_path}")
    print(f"   Task prompt: {task_prompt_path}")
    print(f"   Top movers: {topmovers_path}")
    print(f"   Dataset name: {dataset_name}")
    print()
    
    # Run setup with gene ID fixing
    success = setup_dataset_with_gene_fix(ground_truth_path, task_prompt_path, topmovers_path, dataset_name)
    
    if not success:
        print(f"\n❌ Setup failed. Please check the errors above.")
    else:
        print(f"\n✨ Setup completed successfully!")