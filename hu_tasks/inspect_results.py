import json
import glob
import os

# 1. Find the most recent result file
list_of_files = glob.glob('./eval_results/**/*.json', recursive=True)
if not list_of_files:
    print("No result files found in ./eval_results")
    exit()
latest_file = max(list_of_files, key=os.path.getctime)

print(f"Reading: {latest_file}\n")

with open(latest_file, 'r') as f:
    data = json.load(f)

# 2. Get the samples
task_name = "ocnli_hu" 
if "samples" not in data or task_name not in data["samples"]:
    print("No samples found. Did you use --log_samples?")
    exit()

samples = data["samples"][task_name]
choices = ["蕴含", "矛盾", "中立"] # Your doc_to_choice

# 3. Print a readable table
print(f"{'PREMISE':<20} | {'HYPOTHESIS':<20} | {'TARGET':<10} | {'MODEL PRED':<10} | {'CORRECT'}")
print("-" * 100)

for item in samples:
    # Get raw data
    premise = item['doc']['premise']
    hypothesis = item['doc']['hypothesis']
    
    # Get Target Label
    target_idx = item['target']
    target_str = choices[target_idx]
    
    # Get Model Prediction
    # 'resps' is a list of [log_prob, ...] pairs. We want the index of the max log_prob.
    # Note: Structure of resps might vary slightly by version, usually it's list of floats or list of lists.
    resps = item['resps']
    
    # Extract just the logprobs if nested
    logprobs = [r[0] if isinstance(r, list) else r for r in resps]
    
    # Find index of max value
    pred_idx = logprobs.index(max(logprobs))
    pred_str = choices[pred_idx]
    
    is_correct = "YES" if pred_idx == target_idx else "NO"

    # Truncate text for display
    p_disp = (premise[:18] + '..') if len(premise) > 18 else premise
    h_disp = (hypothesis[:18] + '..') if len(hypothesis) > 18 else hypothesis

    print(f"{p_disp:<20} | {h_disp:<20} | {target_str:<10} | {pred_str:<10} | {is_correct}")