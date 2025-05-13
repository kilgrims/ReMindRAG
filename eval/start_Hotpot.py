import subprocess
import argparse
import os
import time
from datetime import datetime
import json


def run_title_test(title_index, test_name, model_name, question_type):
    cmd = [ "python", "eval_Hotpot.py", 
            "--title_index", str(title_index), 
            "--test_name", test_name,
            "--model_name", model_name,
            "--question_type", question_type
            ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, shell=True)
    return process

def main():
    parser = argparse.ArgumentParser(description='Run ReMindRag Test---Multi-Hop')
    parser.add_argument('--start_index', type=int, default=0, help='Starting title index')
    parser.add_argument('--test_count', type=int, default=97, help='Number of titles to test')
    parser.add_argument('--test_name', type=str, default="test", help='Test name')
    parser.add_argument('--parallel', type=int, default=3, help='Number of parallel tests')
    parser.add_argument('--question_type', type=str, default="origin", choices=["origin", "similar", "different"], help='Question Type: origin, similar or different')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='Backbone model name')
    
    args = parser.parse_args()
    
    start_index = args.start_index
    test_count = args.test_count
    test_name = args.test_name
    parallel = args.parallel
    model_name = args.model_name
    question_type = args.question_type
    
    if parallel < 1:
        parallel = 1
    
    print(f"Starting test...")
    print(f"Test name: {test_name}")
    print(f"Start index: {start_index}")
    print(f"Number of tests: {test_count}")
    print(f"Parallel tests: {parallel}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results_{test_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/config.txt", "w") as f:
        f.write(f"Test name: {test_name}\n")
        f.write(f"Start index: {start_index}\n")
        f.write(f"Number of tests: {test_count}\n")
        f.write(f"Parallel tests: {parallel}\n")
        f.write(f"Start time: {timestamp}\n")
    
    active_processes = []
    results = []
    
    end_index = start_index + test_count
    current_index = start_index
    
    while current_index < end_index or active_processes:
        while current_index < end_index and len(active_processes) < parallel:

            input_file = f"database/{test_name}/{current_index}/input.json"
            if os.path.exists(input_file):
                print(f"Title index {current_index} has already been tested, skipping...")
                current_index += 1
                continue

            process = run_title_test(current_index, test_name, model_name, question_type)
            active_processes.append((process, current_index))
            print(f"Started test for title index {current_index}, PID: {process.pid}")
            current_index += 1
        
        for i in range(len(active_processes) - 1, -1, -1):
            process, index = active_processes[i]
            if process.poll() is not None:
                active_processes.pop(i)
                status = "Success" if process.returncode == 0 else f"Failed (return code: {process.returncode})"
                print(f"Test for title index {index} completed, status: {status}")
                
                try:
                    input_file = f"database/{test_name}/{index}/input.json"
                    
                    if os.path.exists(input_file):
                        with open(input_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            check_response = data.get("check_response", "")
                            is_correct = check_response == "True"
                            
                            results.append({
                                "index": index,
                                "correct": 1 if is_correct else 0,
                                "total": 1,
                                "rate": 1.0 if is_correct else 0.0
                            })
                except Exception as e:
                    print(f"Failed to read results: {e}")
        
        if active_processes:
            time.sleep(2)
    
    total_correct = sum(r["correct"] for r in results)
    total_questions = len(results)
    
    print(f"\nTest completed!")
    print(f"Total correct answers: {total_correct}/{total_questions}")
    print(f"Overall accuracy: {total_correct/total_questions:.4f}" if total_questions > 0 else "No results")
    
    with open(f"{results_dir}/summary.txt", "w") as f:
        f.write(f"Test name: {test_name}\n")
        f.write(f"Total correct answers: {total_correct}/{total_questions}\n")
        if total_questions > 0:
            f.write(f"Overall accuracy: {total_correct/total_questions:.4f}\n\n")
        
        f.write("Results by title:\n")
        for r in results:
            f.write(f"Title {r['index']}: {r['correct']}/{r['total']} = {r['rate']:.4f}\n")
    
    with open(f"{results_dir}/detailed_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
