import sys
sys.path.append('../../')

from datasets import Dataset, load_dataset
from ReMindRag.llms import OpenaiAgent
import json
import random
import os
from tqdm import tqdm

title_start = 0
title_nums = 20

type = "shortdep_qa"
save_dir = f"../dataset_cache/LooGLE-rewrite-data/similar-data/{type}/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir) 


with open('../../api_key.json', 'r', encoding='utf-8') as file:
    api_data = json.load(file)

base_url = api_data[0]["base_url"]
api_key = api_data[0]["api_key"]


ds = load_dataset("bigai-nlco/LooGLE", type, split='test', cache_dir="../dataset_cache") 


agent = OpenaiAgent(base_url, api_key, "gpt-4o")

prompt = """
Now I'll give you a question, and I want you to rewrite it as a different question that means the same thing but is phrased differently.
Original question: {query}

Example:
Original question: Is Beijing the capital of China?
Your Output: Is the capital of the People's Republic of China Beijing?

When you respond, just give me your rewritten question.
"""

all_titles = []
with open("../dataset_cache/LooGLE-rewrite-data/titles.json","r",encoding='utf-8') as f:
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

        input_msg = prompt.format(query = query)
        new_query = agent.generate_response("", [{"role":"user","content":input_msg}])

        cleaned_data.append({"question":new_query})
    
    with open(title_save_dir, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)