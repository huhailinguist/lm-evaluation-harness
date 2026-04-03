import os
import json
import csv
import lm_eval
from lm_eval.tasks import TaskManager

# --- Configuration ---
MODELS = [
    "/root/autodl-tmp/models/Qwen3-0.6B-Base/",
    "/root/autodl-tmp/models/Qwen3-1.7B-Base/",
    "/root/autodl-tmp/models/Qwen3-4B-Base/",
    "/root/autodl-tmp/models/Qwen3-8B-Base/",
    "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-3B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct/",
    # "/root/autodl-tmp/models/Qwen2.5-14B-Instruct-AWQ/",
    # Add other model paths here
]
TASKS = ["ocnli_all"]
INCLUDE_PATH = "/root/autodl-tmp/lm-evaluation-harness/hu_tasks"
OUTPUT_FILE = "/root/autodl-tmp/lm-evaluation-harness/hu_tasks/comparison.csv"
BATCH_SIZE = 64

def evaluate_model(model_path, task_manager):
    """Runs evaluation for a single model and returns the raw results."""
    print(f"Testing: {model_path}...")
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=TASKS,
        batch_size=BATCH_SIZE,
        device="cuda",
        task_manager=task_manager,
        log_samples=True,
    )
    return results

def extract_metrics(model_path, results):
    """Extracts key metrics from the raw results for the CSV."""
    rows = []
    for task_name, metrics in results['results'].items():
        # Create a row with Model Name, Task Name, and all metric values
        row = {"Model": model_path, "Task": task_name}
        # Flatten the metrics dictionary (e.g., {'acc': 0.5} -> row['acc'] = 0.5)
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                row[k] = v
        rows.append(row)
    return rows

def save_to_csv(all_data, filename):
    """Saves the list of dictionaries to a CSV file."""
    if not all_data: return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Get all unique headers dynamically
    headers = sorted(list(set().union(*(d.keys() for d in all_data))))
    
    # Move 'Model' and 'Task' to the front for readability
    for col in ['Task', 'Model']:
        if col in headers:
            headers.insert(0, headers.pop(headers.index(col)))

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_data)
    print(f"Results saved to {filename}")

def main():
    # Initialize TaskManager once to load custom tasks from ./hu_tasks
    tm = TaskManager(include_path=INCLUDE_PATH)
    
    all_results_summary = []

    for model in MODELS:
        # 1. Evaluate
        print(f"\n*****working on {model}")
        raw_results = evaluate_model(model, tm)
        
        # 2. Extract Metrics
        summary_rows = extract_metrics(model, raw_results)
        all_results_summary.extend(summary_rows)

        # 3. Save Full Logs (Optional: preserves the --output_path behavior)
        model_name = os.path.basename(os.path.normpath(model))
        with open(f"./eval_results/{model_name}_full.json", "w") as f:
            json.dump(raw_results, f, indent=2, default=str)

    # 4. Save Comparison CSV
    save_to_csv(all_results_summary, OUTPUT_FILE)

if __name__ == "__main__":
    main()