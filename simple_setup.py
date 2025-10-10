#!/usr/bin/env python3
"""
Fixed setup script for BioDiscoveryAgent.
This version lets you specify the exact file paths.
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

def setup_dataset_with_paths(ground_truth_path, task_prompt_path, topmovers_path, dataset_name="MY_SCREEN"):
    """Set up dataset with specific file paths."""
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
        
        # 3. Process top movers
        print(f"\n🎯 Processing top movers...")
        topmovers = np.load(topmovers_path, allow_pickle=True)
        print(f"   - Loaded {len(topmovers)} top movers")
        print(f"   - Sample: {topmovers[:5].tolist()}")
        
        # 4. Create evaluation ground truth
        print(f"\n⚖️ Creating evaluation ground truth...")
        
        # Positive examples (top movers that exist in the data)
        valid_topmovers = [gene for gene in topmovers if gene in df.index]
        print(f"   - Valid top movers: {len(valid_topmovers)}")
        
        if len(valid_topmovers) == 0:
            print("❌ No top movers found in ground truth data!")
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
        
        # 5. Create metadata
        print(f"\n📋 Creating metadata...")
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
            "source_files": {
                "ground_truth": ground_truth_path,
                "task_prompt": task_prompt_path,
                "topmovers": topmovers_path
            }
        }
        
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

def find_your_files():
    """Help find your original generated files."""
    print("🔍 Searching for your original files...")
    
    # Common search locations
    search_locations = [
        ".",
        "..",
        "datasets",
        "../datasets", 
        os.path.expanduser("~"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop")
    ]
    
    found_files = {
        'ground_truth': [],
        'task_prompt': [],
        'topmovers': []
    }
    
    for location in search_locations:
        if not os.path.exists(location):
            continue
            
        try:
            for root, dirs, files in os.walk(location):
                # Skip hidden directories and common irrelevant directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Look for ground truth files
                    if file.startswith('ground_truth_') and file.endswith('.csv'):
                        found_files['ground_truth'].append(file_path)
                    
                    # Look for task prompt files  
                    elif file.endswith('.json') and ('task' in file.lower() or 'prompt' in file.lower()):
                        found_files['task_prompt'].append(file_path)
                    
                    # Look for topmovers files
                    elif file.startswith('topmovers_') and file.endswith('.npy'):
                        found_files['topmovers'].append(file_path)
                        
        except (PermissionError, OSError):
            continue
    
    print("\n📋 Found files:")
    for file_type, files in found_files.items():
        print(f"\n{file_type.upper()}:")
        if files:
            for i, file_path in enumerate(files):
                print(f"  {i+1}. {file_path}")
        else:
            print("  None found")
    
    return found_files

if __name__ == "__main__":
    print("🔧 BioDiscoveryAgent Dataset Setup")
    print("=" * 50)
    
    # MANUAL SETUP - Replace these paths with your actual file locations
    # You can find your files by running: find ~ -name "ground_truth_*" -o -name "topmovers_*" -o -name "*task*json" 2>/dev/null
    
    print("📝 Manual setup mode - please update the file paths below:")
    print()
    
    # UPDATE THESE PATHS TO YOUR ACTUAL FILES:
    ground_truth_path = '/Users/jordansztejman/Downloads/BioDiscoveryAgent-master/datasets/ground_truth_gene_essentiality_screen.csv'
    task_prompt_path = '/Users/jordansztejman/Downloads/BioDiscoveryAgent-master/datasets/task_prompts/gene_essentiality_screen.json'
    topmovers_path = '/Users/jordansztejman/Downloads/BioDiscoveryAgent-master/datasets/topmovers_gene_essentiality_screen.npy'
    dataset_name = 'MY_CUSTOM_SCREEN'
    
    print("🎯 Current file paths (UPDATE THESE):")
    print(f"   Ground truth: {ground_truth_path}")
    print(f"   Task prompt: {task_prompt_path}")
    print(f"   Top movers: {topmovers_path}")
    print(f"   Dataset name: {dataset_name}")
    print()
    
    # Check if paths are still placeholder values
    if '/path/to/your/' in ground_truth_path:
        print("❌ Please update the file paths in this script!")
        print()
        print("🔍 To find your files, run one of these commands:")
        print("   find ~ -name 'ground_truth_*' 2>/dev/null")
        print("   find ~ -name 'topmovers_*' 2>/dev/null") 
        print("   find ~ -name '*task*.json' 2>/dev/null")
        print()
        print("Then update the paths in this script and run again.")
        
        # Also try to auto-find files
        print("\n🔍 Auto-searching for files...")
        found_files = find_your_files()
        
        if found_files['ground_truth'] or found_files['task_prompt'] or found_files['topmovers']:
            print("\n💡 Found some files you might want to use:")
            if found_files['ground_truth']:
                print("   Ground truth options:")
                for file in found_files['ground_truth'][:3]:  # Show first 3
                    print(f"     {file}")
            if found_files['task_prompt']:
                print("   Task prompt options:")
                for file in found_files['task_prompt'][:3]:
                    print(f"     {file}")
            if found_files['topmovers']:
                print("   Top movers options:")
                for file in found_files['topmovers'][:3]:
                    print(f"     {file}")
            
            print("\nCopy the correct paths to the script and run again!")
        
    else:
        # Paths have been updated, proceed with setup
        success = setup_dataset_with_paths(ground_truth_path, task_prompt_path, topmovers_path, dataset_name)
        
        if not success:
            print(f"\n❌ Setup failed. Please check the file paths and errors above.")