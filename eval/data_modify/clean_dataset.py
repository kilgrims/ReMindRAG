import sys
sys.path.append('../')

from datasets import Dataset, load_dataset
from DynamicRag.llms import OpenaiAgent
import json
import random
import os
from tqdm import tqdm

title_start = 0
title_nums = 20
# type = "longdep_qa"
type = "shortdep_qa"
save_dir = f"./dataset_cache/cleaned_data2/{type}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir) 


with open('../api_key.json', 'r', encoding='utf-8') as file:
    api_data = json.load(file)

base_url = api_data[1]["base_url"]
api_key = api_data[1]["api_key"]


ds = load_dataset("bigai-nlco/LooGLE", type, split='test', cache_dir="./dataset_cache") 


agent = OpenaiAgent(base_url, api_key, "chatgpt-4o-latest")

prompt = """
I'm going to give you a question, the correct answer, and supporting evidence. 
Based on this information, please rewrite the question as a multiple-choice question with four options. 
In the multiple-choice question you create, the correct option should be {new_ans}.  
When creating the incorrect options (distractors), make sure they are plausible based on the question and evidence, so test-takers can’t easily guess the right answer.
Question: {query}  
Correct answer: {ans}  
Supporting evidence: {evidence}

When you output the question, provide only the question and its options (A, B, C, D). Here’s an example of the expected output format:

Example Output:
What is the capital of the United States?
A. New York
B. Washington
C. San Francisco
D. Los Angeles
"""

all_titles = []
with open("titles.json","r",encoding='utf-8') as f:
    title_data = json.load(f)

for title_iter in title_data.values():
    all_titles.append(title_iter)

all_titles = all_titles[title_start:(title_start+title_nums)]



for title in all_titles:
    print(f"Hanlde title: {title}")
    title_save_dir = save_dir + f"{title}.json"
    filtered_data = ds.filter(lambda example: example["title"] == title)
    cleaned_data = []
    for data_iter in tqdm(filtered_data):
        query = data_iter["question"]
        ans = data_iter["answer"]
        evidence = data_iter["evidence"]
        new_ans = random.choice(['A', 'B', 'C', 'D'])

        input_msg = prompt.format(query = query, ans = ans, evidence = evidence, new_ans = new_ans)
        new_query = agent.generate_response("", [{"role":"user","content":input_msg}])

        cleaned_data.append({"question":new_query,"answer":new_ans,"evidence":evidence})
    
    with open(title_save_dir, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)