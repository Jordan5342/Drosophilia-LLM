#!/usr/bin/env python3
"""
Fixed version of gene name conversion that handles existing GENE_ prefixes
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

def clean_and_analyze_current_state():
    """Analyze current state and clean up any issues."""
    
    print("🔍 Analyzing current dataset state...")
    
    # Check main dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    print(f"📊 Main dataset:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Sample genes: {list(df.index[:5])}")
    print(f"  - Gene type: {type(df.index[0])}")
    
    # Check ground truth
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    
    print(f"📊 Ground truth:")
    print(f"  - Shape: {gt_df.shape}")
    print(f"  - Sample genes: {list(gt_df['Gene'][:5])}")
    print(f"  - Gene type: {type(gt_df['Gene'].iloc[0])}")
    
    # Check topmovers
    topmovers_file = Path("datasets/topmovers_MY_CUSTOM_SCREEN.npy")
    if topmovers_file.exists():
        topmovers = np.load(topmovers_file, allow_pickle=True)
        print(f"📊 Topmovers:")
        print(f"  - Count: {len(topmovers)}")
        print(f"  - Sample: {list(topmovers[:5])}")
        print(f"  - Type: {type(topmovers[0])}")
    
    return df, gt_df, topmovers

def restore_from_backups_and_restart():
    """Restore from backups and start fresh."""
    
    print("\n🔄 Restoring from backups and starting fresh...")
    
    # Restore main dataset from backup
    backup_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_numeric_ids.csv")
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    
    if backup_file.exists():
        import shutil
        shutil.copy2(backup_file, main_file)
        print(f"✅ Restored main dataset from backup")
    else:
        print(f"❌ No backup found for main dataset")
        return False
    
    # Restore ground truth from backup
    gt_backup = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth_numeric_ids.csv")
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    
    if gt_backup.exists():
        import shutil
        shutil.copy2(gt_backup, gt_file)
        print(f"✅ Restored ground truth from backup")
    else:
        print(f"❌ No backup found for ground truth")
        return False
    
    # Restore topmovers if backup exists
    topmovers_backup = Path("datasets/topmovers_MY_CUSTOM_SCREEN_numeric_ids.npy")
    topmovers_file = Path("datasets/topmovers_MY_CUSTOM_SCREEN.npy")
    
    if topmovers_backup.exists():
        import shutil
        shutil.copy2(topmovers_backup, topmovers_file)
        print(f"✅ Restored topmovers from backup")
    
    return True

def extract_numeric_id(gene_str):
    """Extract numeric ID from various gene string formats."""
    
    # Handle different formats:
    # "GENE_1234" -> 1234
    # "GENE_GENE_1234" -> 1234  
    # "np.str_('GENE_1234')" -> 1234
    # "1234" -> 1234
    
    gene_str = str(gene_str)
    
    # Remove np.str_() wrapper if present
    if "np.str_(" in gene_str:
        gene_str = gene_str.replace("np.str_(", "").replace(")", "").strip("'\"")
    
    # Extract numbers from the string
    numbers = re.findall(r'\d+', gene_str)
    
    if numbers:
        return int(numbers[-1])  # Take the last number found
    else:
        # If no numbers found, return None
        return None

def create_proper_gene_mapping():
    """Create proper gene mapping from numeric IDs to clean gene names."""
    
    print("\n🔧 Creating proper gene mapping...")
    
    # Read the restored numeric dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    
    print(f"📊 Working with {df.shape[0]} genes")
    print(f"📝 Sample original genes: {list(df.index[:5])}")
    
    # Create clean mapping
    gene_mapping = {}
    for gene_id in df.index:
        # Extract numeric part if it's already a string with GENE_ prefix
        if isinstance(gene_id, str) and "GENE" in gene_id:
            numeric_id = extract_numeric_id(gene_id)
            if numeric_id is not None:
                clean_name = f"GENE_{numeric_id}"
            else:
                clean_name = gene_id  # Keep as-is if we can't extract
        else:
            # It's a numeric ID
            clean_name = f"GENE_{gene_id}"
        
        gene_mapping[gene_id] = clean_name
    
    print(f"✅ Created mapping for {len(gene_mapping)} genes")
    print(f"📝 Sample mappings:")
    for i, (old_id, new_name) in enumerate(list(gene_mapping.items())[:5]):
        print(f"    {old_id} -> {new_name}")
    
    return gene_mapping, df

def apply_clean_gene_mapping(gene_mapping, df):
    """Apply the clean gene mapping to all files."""
    
    print(f"\n🔄 Applying clean gene mapping...")
    
    # 1. Update main dataset
    df_clean = df.copy()
    df_clean.index = [gene_mapping[gene_id] for gene_id in df.index]
    df_clean.index.name = 'Gene'
    
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df_clean.to_csv(main_file)
    print(f"✅ Updated main dataset: {main_file}")
    
    # 2. Update ground truth
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    
    gt_df_clean = gt_df.copy()
    gt_df_clean['Gene'] = gt_df_clean['Gene'].map(lambda x: gene_mapping.get(x, f"GENE_{extract_numeric_id(str(x)) or x}"))
    gt_df_clean.to_csv(gt_file, index=False)
    print(f"✅ Updated ground truth: {gt_file}")
    
    # 3. Update topmovers
    topmovers_file = Path("datasets/topmovers_MY_CUSTOM_SCREEN.npy")
    if topmovers_file.exists():
        try:
            topmovers = np.load(topmovers_file, allow_pickle=True)
            
            topmovers_clean = []
            for gene in topmovers:
                # Try to map using the gene_mapping first
                gene_str = str(gene)
                if gene_str in gene_mapping:
                    topmovers_clean.append(gene_mapping[gene_str])
                else:
                    # Extract numeric ID and create clean name
                    numeric_id = extract_numeric_id(gene_str)
                    if numeric_id is not None:
                        topmovers_clean.append(f"GENE_{numeric_id}")
                    else:
                        # Fallback - use original
                        topmovers_clean.append(gene_str)
            
            np.save(topmovers_file, topmovers_clean)
            print(f"✅ Updated topmovers: {topmovers_file}")
            print(f"📝 Sample topmovers: {topmovers_clean[:5]}")
            
        except Exception as e:
            print(f"❌ Error updating topmovers: {e}")
    
    # 4. Update datasets ground truth
    datasets_gt_file = Path("datasets/ground_truth_MY_CUSTOM_SCREEN.csv")
    if datasets_gt_file.exists():
        try:
            datasets_gt = pd.read_csv(datasets_gt_file)
            datasets_gt_clean = datasets_gt.copy()
            
            if 'Gene' in datasets_gt.columns:
                datasets_gt_clean['Gene'] = datasets_gt_clean['Gene'].map(
                    lambda x: gene_mapping.get(x, f"GENE_{extract_numeric_id(str(x)) or x}")
                )
            
            datasets_gt_clean.to_csv(datasets_gt_file, index=False)
            print(f"✅ Updated datasets ground truth: {datasets_gt_file}")
            
        except Exception as e:
            print(f"❌ Error updating datasets ground truth: {e}")
    
    return df_clean, gt_df_clean

def verify_final_state():
    """Verify that everything is properly formatted."""
    
    print(f"\n🔍 Verifying final state...")
    
    success = True
    
    # Check main dataset
    main_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN.csv")
    df = pd.read_csv(main_file, index_col=0)
    print(f"📊 Main dataset: {df.shape}")
    print(f"   Sample genes: {list(df.index[:5])}")
    
    # Verify all genes have GENE_ prefix
    all_gene_format = all(isinstance(g, str) and g.startswith('GENE_') for g in df.index)
    print(f"   All genes properly formatted: {all_gene_format}")
    if not all_gene_format:
        success = False
    
    # Check ground truth
    gt_file = Path("data/MY_CUSTOM_SCREEN/MY_CUSTOM_SCREEN_ground_truth.csv")
    gt_df = pd.read_csv(gt_file)
    print(f"📊 Ground truth: {gt_df.shape}")
    print(f"   Sample genes: {list(gt_df['Gene'][:5])}")
    
    # Check overlap
    main_genes = set(df.index)
    gt_genes = set(gt_df['Gene'])
    overlap = main_genes.intersection(gt_genes)
    
    print(f"🔄 Gene overlap: {len(overlap)}/{len(gt_genes)} ({len(overlap)/len(gt_genes)*100:.1f}%)")
    
    if len(overlap) == len(gt_genes):
        print(f"✅ Perfect gene overlap!")
    else:
        print(f"⚠️ Gene overlap issue")
        success = False
    
    return success

if __name__ == "__main__":
    print("🔧 Fixed Gene Name Conversion v2")
    print("=" * 45)
    
    try:
        # Step 1: Analyze current state
        df, gt_df, topmovers = clean_and_analyze_current_state()
        
        # Step 2: Restore from backups if needed
        if not restore_from_backups_and_restart():
            print("❌ Could not restore from backups. Please check your files.")
            exit(1)
        
        # Step 3: Create proper mapping
        gene_mapping, df_restored = create_proper_gene_mapping()
        
        # Step 4: Apply clean mapping
        df_final, gt_final = apply_clean_gene_mapping(gene_mapping, df_restored)
        
        # Step 5: Verify everything
        if verify_final_state():
            print(f"\n🎉 SUCCESS! Gene names properly converted!")
            print(f"\n🚀 Now try BioDiscoveryAgent:")
            print(f"python3 research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name test_run --data_name MY_CUSTOM_SCREEN --steps 1 --num_genes 16 --log_dir test_logs")
        else:
            print(f"\n⚠️ Some issues remain. Please check the output above.")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()