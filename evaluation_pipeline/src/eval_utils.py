import json
import os

def load_test_cases(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results, checkpoint, results_dir):
    result_path = os.path.join(results_dir, f"{checkpoint}.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
