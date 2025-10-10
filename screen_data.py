import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class BioDiscoveryAgentDataPrep:
    """
    Prepare custom data for the Stanford BioDiscoveryAgent.
    Converts your formatted data into the required structure for the agent.
    """
    
    def __init__(self, biodiscovery_agent_path: str):
        """
        Initialize with path to the BioDiscoveryAgent repository.
        
        Args:
            biodiscovery_agent_path: Path to the cloned BioDiscoveryAgent repository
        """
        self.agent_path = Path(biodiscovery_agent_path)
        self.data_dir = self.agent_path / "data"
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        print(f"✅ BioDiscoveryAgent path: {self.agent_path}")
        print(f"📁 Data directory: {self.data_dir}")
    
    def convert_screen_data_to_agent_format(
        self, 
        ground_truth_csv: str,
        task_prompt_json: str,
        topmovers_npy: str,
        new_data_name: str
    ) -> Dict[str, str]:
        """
        Convert your formatted screen data to BioDiscoveryAgent format.
        
        Args:
            ground_truth_csv: Path to your ground_truth_*.csv file
            task_prompt_json: Path to your task_prompts/*.json file
            topmovers_npy: Path to your topmovers_*.npy file
            new_data_name: Name for the new dataset (e.g., "MY_SCREEN")
        
        Returns:
            Dictionary with paths to created files
        """
        print(f"\n🔄 Converting data for BioDiscoveryAgent...")
        print(f"📊 Dataset name: {new_data_name}")
        
        # Create directory for this dataset
        dataset_dir = self.data_dir / new_data_name
        dataset_dir.mkdir(exist_ok=True)
        
        created_files = {}
        
        # 1. Load your ground truth data
        print("📋 Loading ground truth data...")
        df = pd.read_csv(ground_truth_csv, index_col=0)
        
        # 2. Load task information
        with open(task_prompt_json, 'r') as f:
            task_info = json.load(f)
        
        # 3. Load top movers
        topmovers = np.load(topmovers_npy, allow_pickle=True)
        
        # 4. Create the main dataset file (similar to IFNG format)
        # The agent expects a CSV with gene names and scores
        output_csv = dataset_dir / f"{new_data_name}.csv"
        
        # Format: Gene,Score (or whatever column name you have)
        agent_df = df.copy()
        agent_df.index.name = 'Gene'  # Ensure index is named 'Gene'
        agent_df.to_csv(output_csv)
        created_files['main_data'] = str(output_csv)
        
        print(f"✅ Created main data file: {output_csv}")
        print(f"   - Shape: {agent_df.shape}")
        print(f"   - Columns: {list(agent_df.columns)}")
        print(f"   - Score range: {agent_df.iloc[:, 0].min():.3f} to {agent_df.iloc[:, 0].max():.3f}")
        
        # 5. Create ground truth file (top hits for evaluation)
        ground_truth_file = dataset_dir / f"{new_data_name}_ground_truth.csv"
        
        # Create ground truth from top movers
        gt_df = pd.DataFrame({
            'Gene': topmovers,
            'Label': 1  # Mark as positive hits
        })
        
        # Add some negative examples (random low-scoring genes)
        low_scoring_genes = df.nsmallest(len(topmovers), df.columns[0]).index
        negative_df = pd.DataFrame({
            'Gene': low_scoring_genes,
            'Label': 0  # Mark as negative
        })
        
        # Combine positive and negative
        full_gt_df = pd.concat([gt_df, negative_df], ignore_index=True)
        full_gt_df.to_csv(ground_truth_file, index=False)
        created_files['ground_truth'] = str(ground_truth_file)
        
        print(f"✅ Created ground truth file: {ground_truth_file}")
        print(f"   - Positive hits: {len(topmovers)}")
        print(f"   - Negative examples: {len(low_scoring_genes)}")
        
        # 6. Create metadata file
        metadata_file = dataset_dir / f"{new_data_name}_metadata.json"
        
        metadata = {
            "dataset_name": new_data_name,
            "description": task_info.get('Task', f'Custom dataset: {new_data_name}'),
            "measurement": task_info.get('Measurement', 'Gene importance score'),
            "total_genes": len(df),
            "positive_hits": len(topmovers),
            "score_stats": {
                "min": float(df.iloc[:, 0].min()),
                "max": float(df.iloc[:, 0].max()),
                "mean": float(df.iloc[:, 0].mean()),
                "std": float(df.iloc[:, 0].std())
            },
            "top_genes": topmovers[:10].tolist(),
            "created_from": {
                "ground_truth_csv": ground_truth_csv,
                "task_prompt_json": task_prompt_json,
                "topmovers_npy": topmovers_npy
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        created_files['metadata'] = str(metadata_file)
        
        print(f"✅ Created metadata file: {metadata_file}")
        
        # 7. Update the main data registry (if it exists)
        self._update_data_registry(new_data_name, metadata)
        
        return created_files
    
    def _update_data_registry(self, data_name: str, metadata: Dict) -> None:
        """Update or create a data registry file."""
        registry_file = self.data_dir / "data_registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"datasets": {}}
        
        registry["datasets"][data_name] = {
            "path": f"data/{data_name}/{data_name}.csv",
            "ground_truth": f"data/{data_name}/{data_name}_ground_truth.csv",
            "metadata": f"data/{data_name}/{data_name}_metadata.json",
            "description": metadata["description"],
            "total_genes": metadata["total_genes"],
            "positive_hits": metadata["positive_hits"]
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Updated data registry: {registry_file}")
    
    def create_run_script(self, data_name: str, output_path: str = None) -> str:
        """
        Create a shell script to run the BioDiscoveryAgent with your data.
        
        Args:
            data_name: Name of your dataset
            output_path: Path to save the script (optional)
        
        Returns:
            Path to the created script
        """
        if output_path is None:
            output_path = self.agent_path / f"run_{data_name.lower()}.sh"
        
        script_content = f"""#!/bin/bash

# Run BioDiscoveryAgent with {data_name} dataset
# Generated automatically

echo "🚀 Running BioDiscoveryAgent with {data_name} dataset"

# Basic run (no additional tools)
echo "📊 Basic run..."
python research_assistant.py \\
    --task perturb-genes-brief \\
    --model claude-3-5-sonnet-20240620 \\
    --run_name {data_name.lower()}_basic \\
    --data_name {data_name} \\
    --steps 5 \\
    --num_genes 64 \\
    --log_dir {data_name.lower()}_logs

echo "✅ Basic run completed"

# Run with all tools (if you want the full experience)
echo "🔧 Run with all tools..."
python research_assistant.py \\
    --task perturb-genes-brief \\
    --model claude-3-5-sonnet-20240620 \\
    --run_name {data_name.lower()}_full \\
    --data_name {data_name} \\
    --steps 5 \\
    --num_genes 64 \\
    --log_dir {data_name.lower()}_logs_full \\
    --lit_review True \\
    --critique True \\
    --reactome True

echo "✅ Full run completed"

# Analyze results
echo "📈 Analyzing results..."
python analyze.py \\
    --model claude-3-5-sonnet-20240620_all \\
    --dataset {data_name} \\
    --rounds 5

echo "🎉 All done! Check the log directories for results."
"""
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        
        print(f"✅ Created run script: {output_path}")
        print(f"   To execute: bash {output_path}")
        
        return str(output_path)
    
    def validate_setup(self, data_name: str) -> bool:
        """
        Validate that the data has been set up correctly for the agent.
        
        Args:
            data_name: Name of your dataset
        
        Returns:
            True if setup is valid
        """
        print(f"\n🔍 Validating setup for {data_name}...")
        
        dataset_dir = self.data_dir / data_name
        
        # Check required files
        required_files = [
            dataset_dir / f"{data_name}.csv",
            dataset_dir / f"{data_name}_ground_truth.csv",
            dataset_dir / f"{data_name}_metadata.json"
        ]
        
        all_valid = True
        
        for file_path in required_files:
            if file_path.exists():
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                all_valid = False
        
        # Check main agent files
        agent_files = [
            self.agent_path / "research_assistant.py",
            self.agent_path / "analyze.py"
        ]
        
        for file_path in agent_files:
            if file_path.exists():
                print(f"✅ Found: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
                all_valid = False
        
        if all_valid:
            print(f"🎉 Setup validation passed! Ready to run BioDiscoveryAgent with {data_name}")
        else:
            print(f"❌ Setup validation failed. Please check missing files.")
        
        return all_valid
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets in the agent's data directory."""
        datasets = []
        
        if not self.data_dir.exists():
            return datasets
        
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Check if it has the required files
                main_file = item / f"{item.name}.csv"
                if main_file.exists():
                    datasets.append(item.name)
        
        return datasets


def setup_custom_dataset_for_agent(
    biodiscovery_agent_path: str,
    ground_truth_csv: str,
    task_prompt_json: str,
    topmovers_npy: str,
    new_data_name: str
) -> Dict[str, str]:
    """
    Complete setup of custom dataset for BioDiscoveryAgent.
    
    Args:
        biodiscovery_agent_path: Path to BioDiscoveryAgent repository
        ground_truth_csv: Path to your ground_truth_*.csv
        task_prompt_json: Path to your task_prompt_*.json
        topmovers_npy: Path to your topmovers_*.npy
        new_data_name: Name for your dataset (e.g., "MY_SCREEN")
    
    Returns:
        Dictionary with information about created files
    """
    print("🔧 Setting up custom dataset for BioDiscoveryAgent...")
    
    # Initialize the data prep class
    prep = BioDiscoveryAgentDataPrep(biodiscovery_agent_path)
    
    # Convert data to agent format
    created_files = prep.convert_screen_data_to_agent_format(
        ground_truth_csv, task_prompt_json, topmovers_npy, new_data_name
    )
    
    # Create run script
    script_path = prep.create_run_script(new_data_name)
    created_files['run_script'] = script_path
    
    # Validate setup
    is_valid = prep.validate_setup(new_data_name)
    
    # Show available datasets
    datasets = prep.list_available_datasets()
    print(f"\n📚 Available datasets: {datasets}")
    
    return {
        **created_files,
        'is_valid': is_valid,
        'available_datasets': datasets,
        'run_command': f"python research_assistant.py --task perturb-genes-brief --model claude-3-5-sonnet-20240620 --run_name {new_data_name.lower()}_demo --data_name {new_data_name} --steps 3 --num_genes 64 --log_dir {new_data_name.lower()}_logs"
    }


if __name__ == "__main__":
    # Automatic path detection and setup
    import os
    
    # Get current directory (should be BioDiscoveryAgent directory)
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Use current directory as agent path
    agent_path = current_dir
    
    # Auto-detect your generated files
    possible_paths = [
        # Look in current directory
        "datasets/ground_truth_gene_essentiality_screen.csv",
        "datasets/task_prompts/gene_essentiality_screen.json", 
        "datasets/topmovers_gene_essentiality_screen.npy",
        # Look one level up
        "../datasets/ground_truth_gene_essentiality_screen.csv",
        "../datasets/task_prompts/gene_essentiality_screen.json",
        "../datasets/topmovers_gene_essentiality_screen.npy"
    ]
    
    # Find your files
    ground_truth = None
    task_prompt = None
    topmovers = None
    
    for path in possible_paths:
        if os.path.exists(path):
            if "ground_truth" in path:
                ground_truth = path
            elif "task_prompts" in path:
                task_prompt = path
            elif "topmovers" in path:
                topmovers = path
    
    print(f"\n🔍 Auto-detected files:")
    print(f"Ground truth: {ground_truth}")
    print(f"Task prompt: {task_prompt}")
    print(f"Top movers: {topmovers}")
    
    # Check if all files were found
    if not all([ground_truth, task_prompt, topmovers]):
        print("❌ Could not find all required files. Please check:")
        print("   - datasets/ground_truth_*.csv")
        print("   - datasets/task_prompts/*.json")
        print("   - datasets/topmovers_*.npy")
        print("\nOr manually specify the paths:")
        print("ground_truth = 'path/to/your/ground_truth_file.csv'")
        print("task_prompt = 'path/to/your/task_prompt_file.json'")
        print("topmovers = 'path/to/your/topmovers_file.npy'")
        exit(1)
    
    data_name = "MY_CUSTOM_SCREEN"
    
    # Set up the dataset
    try:
        result = setup_custom_dataset_for_agent(
            agent_path, ground_truth, task_prompt, topmovers, data_name
        )
        
        print(f"\n🎯 Ready to run:")
        print(f"cd {agent_path}")
        print(result['run_command'])
        
    except Exception as e:
        print(f"❌ Error setting up dataset: {e}")
        print("\nManual setup option:")
        print(f"agent_path = '{agent_path}'")
        print(f"ground_truth = '{ground_truth}'")
        print(f"task_prompt = '{task_prompt}'")
        print(f"topmovers = '{topmovers}'")
        print(f"data_name = '{data_name}'")
        print("\nThen run:")
        print("result = setup_custom_dataset_for_agent(agent_path, ground_truth, task_prompt, topmovers, data_name)")