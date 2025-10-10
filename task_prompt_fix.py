#!/usr/bin/env python3
"""
Create the missing task prompt file for MY_CUSTOM_SCREEN
"""

import json
import os
from pathlib import Path

def create_task_prompt():
    """Create the task prompt file that BioDiscoveryAgent expects."""
    
    # Create the task prompt directory if it doesn't exist
    prompt_dir = Path("datasets/task_prompts")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    # Task prompt content based on your original gene essentiality screen
    task_prompt = {
        "Task": "Identify genes important for gene_essentiality_screen based on screening data",
        "Measurement": "Gene importance score derived from construct frequency and biological relevance"
    }
    
    # Save the task prompt file
    prompt_file = prompt_dir / "MY_CUSTOM_SCREEN.json"
    with open(prompt_file, 'w') as f:
        json.dump(task_prompt, f, indent=2)
    
    print(f"✅ Created task prompt file: {prompt_file}")
    
    # Verify it was created correctly
    if prompt_file.exists():
        print(f"✅ File exists and is readable")
        with open(prompt_file, 'r') as f:
            content = json.load(f)
        print(f"📝 Content: {content}")
        return True
    else:
        print(f"❌ Failed to create file")
        return False

if __name__ == "__main__":
    print("🔧 Creating missing task prompt file...")
    success = create_task_prompt()
    if success:
        print("✅ Task prompt file created successfully!")
    else:
        print("❌ Failed to create task prompt file")
