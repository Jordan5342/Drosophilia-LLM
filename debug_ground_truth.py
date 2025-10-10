#!/usr/bin/env python3
"""
Create any missing files that BioDiscoveryAgent might expect
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def create_missing_files():
    """Create files that BioDiscoveryAgent might be expecting."""
    
    print("🔧 Creating potentially missing files...")
    
    # 1. Create topmovers file in datasets/ if it doesn't exist
    datasets_dir = Path("datasets")
    topmovers_file = datasets_dir / "topmovers_MY_CUSTOM_SCREEN.npy"
    
    if not topmovers_file.exists():
        print(f"📄 Creating {topmovers_file}...")
        
        # Read ground truth to get positive genes
        gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
        if gt_file.exists():
            gt_df = pd.read_csv(gt_file)
            positive_genes = gt_df[gt_df['Label'] == 1]['Gene'].values
            
            # Save as numpy array (like other topmovers files)
            np.save(topmovers_file, positive_genes)
            print(f"✅ Created {topmovers_file} with {len(positive_genes)} genes")
        else:
            print(f"❌ Cannot create topmovers file - ground truth missing")
    
    # 2. Create ground truth file in datasets/ directory
    datasets_gt_file = datasets_dir / "ground_truth_MY_CUSTOM_SCREEN.csv"
    
    if not datasets_gt_file.exists():
        print(f"📄 Creating {datasets_gt_file}...")
        
        # Copy from data directory
        source_gt = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
        if source_gt.exists():
            gt_df = pd.read_csv(source_gt)
            
            # Check format of other ground truth files
            other_gt_files = list(datasets_dir.glob("ground_truth_*.csv"))
            if other_gt_files:
                # Look at the format of an existing ground truth file
                sample_gt = pd.read_csv(other_gt_files[0], nrows=5)
                print(f"📄 Sample existing ground truth format:\n{sample_gt}")
                print(f"   Columns: {list(sample_gt.columns)}")
            
            # Save in datasets directory
            gt_df.to_csv(datasets_gt_file, index=False)
            print(f"✅ Created {datasets_gt_file}")
        else:
            print(f"❌ Cannot create datasets ground truth - source missing")
    
    # 3. Create CEGv2.txt file if analyze.py needs it
    ceg_file = Path("CEGv2.txt")
    if not ceg_file.exists():
        print(f"📄 Creating {ceg_file} (essential genes file)...")
        
        # Create a simple essential genes file
        # Use genes from your ground truth as a starting point
        gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
        if gt_file.exists():
            gt_df = pd.read_csv(gt_file)
            essential_genes = gt_df[gt_df['Label'] == 1]['Gene'].values
            
            # Create CEGv2.txt format (tab-delimited with GENE column)
            ceg_df = pd.DataFrame({'GENE': essential_genes})
            ceg_df.to_csv(ceg_file, sep='\t', index=False)
            print(f"✅ Created {ceg_file} with {len(essential_genes)} essential genes")
    
    # 4. Check if there are other expected files by looking at existing datasets
    print(f"\n🔍 Checking what files other datasets have...")
    
    data_dir = Path("data")
    if data_dir.exists():
        for dataset_dir in data_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name != "MY_CUSTOM_SCREEN":
                print(f"\n📂 {dataset_dir.name}:")
                for file in dataset_dir.iterdir():
                    if file.is_file():
                        print(f"  📄 {file.name}")
    
    # 5. Create any missing metadata files
    custom_dir = Path("data/MY_CUSTOM_SCREEN")
    metadata_file = custom_dir / "MY_CUSTOM_SCREEN_metadata.json"
    
    if metadata_file.exists():
        print(f"✅ Metadata file exists")
    else:
        print(f"📄 Creating metadata file...")
        
        metadata = {
            "dataset_name": "MY_CUSTOM_SCREEN",
            "description": "Custom genetic screen dataset",
            "total_genes": 15073,  # From your debug output
            "source": "gene_essentiality_screen"
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Created {metadata_file}")

def verify_file_structure():
    """Verify the complete file structure."""
    
    print(f"\n🔍 Verifying complete file structure...")
    
    expected_files = [
        "data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv",
        "data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv",
        "datasets/topmovers_MY_CUSTOM_SCREEN.npy",
        "datasets/ground_truth_MY_CUSTOM_SCREEN.csv",
        "datasets/task_prompts/MY_CUSTOM_SCREEN.json"
    ]
    
    print(f"📋 Checking expected files:")
    all_good = True
    
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} MISSING")
            all_good = False
    
    if all_good:
        print(f"\n✅ All expected files are present!")
        return True
    else:
        print(f"\n⚠️  Some files are missing")
        return False

if __name__ == "__main__":
    print("🔧 Creating Missing Files for BioDiscoveryAgent")
    print("=" * 55)
    
    create_missing_files()
    
    if verify_file_structure():
        print(f"\n🚀 Ready to try BioDiscoveryAgent again!")
        print(f"   python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name my_custom_screen_run --data_name MY_CUSTOM_SCREEN --steps 3 --num_genes 64 --log_dir my_custom_screen_logs")
    else:
        print(f"\n⚠️  Please run the file structure verification again after addressing missing files")