import os
import json
import csv
import torch
import lm_eval
from lm_eval.tasks import TaskManager

# --- Configuration ---
MODELS = [
    "/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-3B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct/",
    "/root/autodl-tmp/models/Qwen2.5-0.5B/",
    "/root/autodl-tmp/models/Qwen2.5-1.5B/",
    "/root/autodl-tmp/models/Qwen2.5-3B/",
    "/root/autodl-tmp/models/Qwen2.5-7B/",
]
TASKS = ["ocnli_300_samples", "ocnli_300_samples2", "ocnli_300_samples3"]
INCLUDE_PATH = "/root/autodl-tmp/lm-evaluation-harness/hu_tasks"
OUTPUT_DIR = "/root/autodl-tmp/lm-evaluation-harness/hu_tasks/eval_res_compare_prompts"
BATCH_SIZE = 32

# --- NEW: map each task name to a human-readable prompt style label ---
TASK_TO_PROMPT_STYLE = {
    "ocnli_300_samples":  "A: 推理关系 (一定能/不一定/不可能)",
    "ocnli_300_samples2": "B: 是否成立 (一定成立/有可能成立/不可能成立)",
    "ocnli_300_samples3": "C: 正确错误 (正确/都有可能/错误)",
}


def evaluate_model(model_path, task_manager):
    """Runs evaluation."""
    print(f"\n\n--- Starting Evaluation: {model_path} ---")

    model_args = f"pretrained={model_path}"

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=TASKS,
        batch_size=BATCH_SIZE,
        device="cuda",
        task_manager=task_manager,
        log_samples=True,
    )
    return results


def extract_metrics(model_path, results):
    """Extracts scalar metrics (acc, loss, etc.) for the summary CSV."""
    rows = []
    if "results" in results:
        for task_name, metrics in results["results"].items():
            row = {
                "Model": model_path,
                "Task": task_name,
                "Prompt_Style": TASK_TO_PROMPT_STYLE.get(task_name, "unknown"),
            }
            for k, v in metrics.items():
                if isinstance(v, (int, float, str)):
                    row[k] = v
            rows.append(row)
    return rows


def save_model_samples(model_path, results, output_dir):
    """Extracts the 'samples' key and saves detailed logs."""
    if "samples" not in results:
        print(f"Warning: No samples found for {model_path}. Did you set log_samples=True?")
        return

    safe_name = model_path.replace("/", "_").replace("\\", "_")
    filename = os.path.join(output_dir, f"{safe_name}_samples.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results["samples"], f, indent=2, default=str, ensure_ascii=False)

    print(f"Detailed samples saved to: {filename}")


def save_summary_csv(all_data, filename):
    """Saves the high-level metrics to a CSV file."""
    if not all_data:
        return

    headers = sorted(list(set().union(*(d.keys() for d in all_data))))
    # Move key columns to the front for readability
    for col in ["Prompt_Style", "Task", "Model"]:
        if col in headers:
            headers.insert(0, headers.pop(headers.index(col)))

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_data)
    print(f"Summary metrics saved to: {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tm = TaskManager(include_path=INCLUDE_PATH)

    all_results_summary = []

    for model in MODELS:
        raw_results = evaluate_model(model, tm)

        summary_rows = extract_metrics(model, raw_results)
        all_results_summary.extend(summary_rows)

        save_model_samples(model, raw_results, OUTPUT_DIR)

        safe_name = model.replace("/", "_")
        with open(os.path.join(OUTPUT_DIR, f"{safe_name}_full_raw.json"), "w") as f:
            json.dump(raw_results, f, indent=2, default=str)

        torch.cuda.empty_cache()

    save_summary_csv(all_results_summary, os.path.join(OUTPUT_DIR, "leaderboard.csv"))


if __name__ == "__main__":
    main()