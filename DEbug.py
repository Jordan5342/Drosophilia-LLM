import json
from pathlib import Path

# Assuming ScreenData class code is already defined above or imported

def prepare_biodiscovery_dataset(
    raw_file_path: str,
    id_col: str,
    val_col: str,
    bio_taskname: str,
    task_prompt: dict
):
    # Step 1: Load and save cleaned ground truth CSV
    sd = ScreenData(
        file_path=raw_file_path,
        id_col=id_col,
        val_col=val_col,
        bio_taskname=bio_taskname,
        save=True
    )
    print(f"Saved cleaned ground truth to datasets/ground_truth_{bio_taskname}.csv")

    # Step 2: Save task prompt JSON file
    base_dir = Path.cwd() / 'datasets'
    prompts_dir = base_dir / 'task_prompts'
    prompts_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompts_dir / f"{bio_taskname}.json"
    with open(prompt_path, 'w') as f:
        json.dump(task_prompt, f, indent=4)
    print(f"Saved task prompt JSON to {prompt_path}")


if __name__ == "__main__":
    # Customize these paths and values:
    raw_file = 'raw_FLY_IMMUNE.csv'  # your raw CSV path
    bio_task = 'FLY_IMMUNE'          # base name for files

    # Your original column names in the raw CSV:
    original_id_col = 'Gene'
    original_val_col = 'Score'

    # Example task prompt dictionary (update as needed)
    task_prompt_example = {
        "Task": "Summarize the effect of the given gene on Drosophila's innate immune response based on the provided context and supporting literature.",
        "Measurement": "The answer should be a concise, factual summary (1-2 sentences) of the gene's perturbation effect."
    }

    prepare_biodiscovery_dataset(
        raw_file_path=raw_file,
        id_col=original_id_col,
        val_col=original_val_col,
        bio_taskname=bio_task,
        task_prompt=task_prompt_example
    )
