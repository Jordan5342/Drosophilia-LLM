#!/usr/bin/env python3
"""
Debug why BioDiscoveryAgent is predicting 0 new genes
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def analyze_dataset_for_prediction_issues():
    """Analyze the dataset to understand why predictions might be failing."""
    
    print("🔍 Analyzing dataset for prediction issues...")
    
    # 1. Check the main dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    if main_file.exists():
        df = pd.read_csv(main_file, index_col=0)
        print(f"\n📊 Main dataset analysis:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Gene ID type: {type(df.index[0])}")
        print(f"  - Score column: {df.columns[0]}")
        print(f"  - Score range: {df[df.columns[0]].min():.6f} to {df[df.columns[0]].max():.6f}")
        print(f"  - Score distribution:")
        print(f"    - Mean: {df[df.columns[0]].mean():.6f}")
        print(f"    - Std: {df[df.columns[0]].std():.6f}")
        print(f"    - Median: {df[df.columns[0]].median():.6f}")
        
        # Check for any obvious issues
        print(f"\n🔍 Data quality checks:")
        print(f"  - Missing values: {df.isnull().sum().sum()}")
        print(f"  - Duplicate genes: {df.index.duplicated().sum()}")
        print(f"  - Zero scores: {(df[df.columns[0]] == 0).sum()}")
        print(f"  - Infinite values: {np.isinf(df[df.columns[0]]).sum()}")
        
        # Show top and bottom genes
        print(f"\n🔝 Top 10 scoring genes:")
        top_genes = df.nlargest(10, df.columns[0])
        for gene, score in top_genes.iterrows():
            print(f"    {gene}: {score[0]:.6f}")
        
        print(f"\n🔻 Bottom 10 scoring genes:")
        bottom_genes = df.nsmallest(10, df.columns[0])
        for gene, score in bottom_genes.iterrows():
            print(f"    {gene}: {score[0]:.6f}")
    
    # 2. Check ground truth
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    if gt_file.exists():
        gt_df = pd.read_csv(gt_file)
        print(f"\n📊 Ground truth analysis:")
        print(f"  - Shape: {gt_df.shape}")
        print(f"  - Positive examples: {(gt_df['Label'] == 1).sum()}")
        print(f"  - Negative examples: {(gt_df['Label'] == 0).sum()}")
        print(f"  - Gene type: {type(gt_df['Gene'].iloc[0])}")
        
        # Check overlap with main dataset
        main_genes = set(str(g) for g in df.index)
        gt_genes = set(str(g) for g in gt_df['Gene'])
        overlap = main_genes.intersection(gt_genes)
        
        print(f"\n🔄 Gene overlap:")
        print(f"  - Main dataset genes: {len(main_genes)}")
        print(f"  - Ground truth genes: {len(gt_genes)}")
        print(f"  - Overlap: {len(overlap)} genes")
        print(f"  - Overlap percentage: {len(overlap)/len(gt_genes)*100:.1f}%")
        
        if len(overlap) < len(gt_genes):
            missing_genes = gt_genes - main_genes
            print(f"  - Missing from main dataset: {len(missing_genes)} genes")
            print(f"    Sample missing: {list(missing_genes)[:5]}")

def check_log_files():
    """Check the log files for more detailed error information."""
    
    print(f"\n🔍 Checking log files...")
    
    # Look for log directories
    log_dirs = []
    for item in Path(".").iterdir():
        if item.is_dir() and "log" in item.name.lower():
            log_dirs.append(item)
    
    if not log_dirs:
        print("❌ No log directories found")
        return
    
    print(f"📁 Found log directories: {[d.name for d in log_dirs]}")
    
    # Check the most recent log directory
    latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📄 Checking latest log directory: {latest_log_dir}")
    
    # Look for relevant log files
    for log_file in latest_log_dir.rglob("*.log"):
        print(f"\n📄 Log file: {log_file}")
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                
                # Show last 20 lines
                print("📝 Last 20 lines:")
                for line in lines[-20:]:
                    if line.strip():
                        print(f"    {line}")
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
    
    # Check for JSON files with detailed info
    for json_file in latest_log_dir.rglob("*.json"):
        print(f"\n📄 JSON file: {json_file}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print(f"📝 Keys: {list(data.keys())}")
                    # Show recent actions if available
                    if 'actions' in data:
                        print(f"📝 Recent actions: {len(data['actions'])}")
                        if data['actions']:
                            print(f"    Last action: {data['actions'][-1]}")
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")

def check_model_configuration():
    """Check if there are issues with model configuration."""
    
    print(f"\n🔍 Checking model configuration...")
    
    # Check if API key exists
    api_files = ["anthropic_api_key.txt", "openai_api_key.txt"]
    for api_file in api_files:
        if Path(api_file).exists():
            print(f"✅ {api_file} exists")
            # Check if it's not empty
            with open(api_file, 'r') as f:
                content = f.read().strip()
                if content:
                    print(f"  - Has content (length: {len(content)})")
                else:
                    print(f"  - ⚠️ File is empty!")
        else:
            print(f"❌ {api_file} missing")

def suggest_fixes():
    """Suggest potential fixes based on the analysis."""
    
    print(f"\n💡 Potential fixes to try:")
    
    print(f"\n1. **Check gene ID consistency:**")
    print(f"   - Make sure all gene IDs are the same type (string vs int)")
    print(f"   - Verify ground truth genes exist in main dataset")
    
    print(f"\n2. **Verify data ranges:**")
    print(f"   - Check if scores are in expected range (0-1, log scale, etc.)")
    print(f"   - Ensure no infinite or NaN values")
    
    print(f"\n3. **Model configuration:**")
    print(f"   - Verify API key is working")
    print(f"   - Check if model is responding properly")
    
    print(f"\n4. **Reduce complexity for testing:**")
    print(f"   - Try with fewer genes: --num_genes 16")
    print(f"   - Try fewer steps: --steps 1")
    print(f"   - Use simpler task if available")
    
    print(f"\n5. **Check BioDiscoveryAgent version:**")
    print(f"   - Make sure you're using a compatible model")
    print(f"   - Check if claude-3-5-sonnet-20240620 is supported")

def create_minimal_test_dataset():
    """Create a minimal test dataset for debugging."""
    
    print(f"\n🔧 Creating minimal test dataset...")
    
    # Create a very simple dataset with clear patterns
    test_genes = [str(i) for i in range(1, 101)]  # 100 genes
    scores = np.random.rand(100)  # Random scores 0-1
    
    # Make sure some genes have very high scores (obvious hits)
    scores[:10] = np.linspace(0.9, 1.0, 10)  # Top 10 genes have high scores
    scores[-10:] = np.linspace(0.0, 0.1, 10)  # Bottom 10 genes have low scores
    
    # Create minimal dataset
    test_df = pd.DataFrame({
        'Score': scores
    }, index=test_genes)
    
    # Create test directory
    test_dir = Path("data/TEST_MINIMAL")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    test_df.to_csv(test_dir / "TEST_MINIMAL.csv")
    
    # Create ground truth (top 10 as positive, bottom 10 as negative)
    gt_df = pd.DataFrame({
        'Gene': [str(i) for i in range(1, 11)] + [str(i) for i in range(91, 101)],
        'Label': [1] * 10 + [0] * 10
    })
    gt_df.to_csv(test_dir / "TEST_MINIMAL_ground_truth.csv", index=False)
    
    # Create files in datasets/
    datasets_dir = Path("datasets")
    
    # Topmovers
    np.save(datasets_dir / "topmovers_TEST_MINIMAL.npy", [str(i) for i in range(1, 11)])
    
    # Ground truth
    gt_df.to_csv(datasets_dir / "ground_truth_TEST_MINIMAL.csv", index=False)
    
    # Task prompt
    task_prompt = {
        "Task": "Identify important genes for test dataset",
        "Measurement": "Gene importance score"
    }
    
    prompt_dir = datasets_dir / "task_prompts"
    prompt_dir.mkdir(exist_ok=True)
    with open(prompt_dir / "TEST_MINIMAL.json", 'w') as f:
        json.dump(task_prompt, f, indent=2)
    
    print(f"✅ Created minimal test dataset: TEST_MINIMAL")
    print(f"   - 100 genes with clear score patterns")
    print(f"   - Top 10 genes as positive hits")
    print(f"   - Bottom 10 genes as negative examples")
    
    print(f"\n🧪 Test command:")
    print(f"python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name test_minimal_run --data_name TEST_MINIMAL --steps 1 --num_genes 16 --log_dir test_minimal_logs")

if __name__ == "__main__":
    print("🔧 Debugging Gene Prediction Issues")
    print("=" * 50)
    
    analyze_dataset_for_prediction_issues()
    check_log_files()
    check_model_configuration()
    suggest_fixes()
    create_minimal_test_dataset()
