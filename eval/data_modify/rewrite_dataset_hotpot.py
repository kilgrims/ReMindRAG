import sys
sys.path.append('../../')

from datasets import Dataset, load_dataset
from ReMindRag.llms import OpenaiAgent
import json
import random
import os
from tqdm import tqdm

title_start = 0
title_nums = 100
type = "different"


save_dir = f"./dataset_cache/Hotpot/hotpot_dev_distractor_{type}.json"


with open('../api_key.json', 'r', encoding='utf-8') as file:
    api_data = json.load(file)

base_url = api_data[0]["base_url"]
api_key = api_data[0]["api_key"]


agent = OpenaiAgent(base_url, api_key, "gpt-4o")

prompt = {}

prompt["similar"]="""
Now I'll give you a question, and I want you to rewrite it as a different question that means the same thing but is phrased differently.
Original question: {query}


Example:
Original question: Is Beijing the capital of China?
Your Output: Is the capital of the People's Republic of China Beijing?

When you respond, just give me your rewritten question.
"""

prompt["different"]="""
Now I'll give you a question, and I want you to rephrase it in a way that remains as similar as possible to the original but may yield a different answer.
Please note that your rephrased question must be answerable based on the reference material.
Original question: {query}
Reference Material: {context}

Example:
Original question: Are both Beijing and Shanghai cities in China?
Your output: Are both Beijing and Chengdu cities in China?

When responding, just provide your rephrased question.
"""

prompt["different_ans"]="""
Now I'll give you a question, and I want you to answer it based on the reference material.
Please note that your answer must come from the reference material.
Question: {query}
Reference Material: {context}

Example:
Original question: Are both Beijing and Shanghai cities in China?
Your output: Yes

When responding, just provide your answer.
"""

with open('./dataset_cache/Hotpot/hotpot_dev_distractor_v1.json', 'r', encoding='utf-8') as file:
    origin_data = json.load(file)

cleaned_data = []

for title in tqdm(range(title_start,title_start+title_nums)):
        query = origin_data[title]["question"]
        
        context = str(origin_data[title]["context"])

        if type == "similar":
            input_msg = prompt[type].format(query = query)
            new_query = agent.generate_response("", [{"role":"user","content":input_msg}])
            anwser = origin_data[title]["answer"]
        else:
            input_msg = prompt[type].format(query = query, context = context)
            new_query = agent.generate_response("", [{"role":"user","content":input_msg}])
            input_msg = prompt["different_ans"].format(query = new_query, context = context)
            anwser = agent.generate_response("", [{"role":"user","content":input_msg}])

        cleaned_data.append({"question":new_query,"answer":anwser})
    
with open(save_dir, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)