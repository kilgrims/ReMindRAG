import sys
sys.path.append('../')

from datasets import load_dataset
from ReMindRag.llms import OpenaiAgent
from ReMindRag.embeddings import HgEmbedding
from ReMindRag.chunking import MetaChunker, NaiveChunker
from ReMindRag import ReMindRag

import json
import torch
from datetime import datetime
import logging
import os
import argparse
from transformers import AutoTokenizer


dataset_name = "hotpot_dev_distractor_v1.json"

def main():
    parser = argparse.ArgumentParser(description='Run ReMindRag Test---Multi-Hop')
    parser.add_argument('--title_index', type=int, required=True, help='Test Index')
    parser.add_argument('--test_name', type=str, default="test", help='Test Name')
    parser.add_argument('--question_type', type=str, choices=["origin", "similar", "different"], help='Question Type: origin, similar or different')
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini", help='Backbone Model Name')
    args = parser.parse_args()

    title_index = args.title_index
    test_name = args.test_name
    model_name = args.model_name
    query_type = args.question_type

    right_num = 0
    total_num = 0

    response_format = """
Origin Query: {query}
LLM Output: {output}

Reference Chunks: {chunks}
Reference Edges: {edges}

"""

    ans_checck_prompt = """
Given one question, there is a groundtruth and a predict answer. 
Please decide whether they are the same or not in semantic. 
Please only output True or False. 
Question: {question}  
groundtruth = {reference_answer}  
predicted answer = {generated_output}

Only output one word(True or False), without any additional content.
"""

    ans_rewrite_prompt = """
Please extract the answer part from the given input. The given input may contain irrelevant content. Please ignore this irrelevant information and only output the answer to the question within the input itself. Ensure that the answer content is exactly the same as the answer in the original text, without any modifications, additions, or deletions.

Original text: {og_output}
Question: {question}
Answer:
"""


    with open('../api_key.json', 'r', encoding='utf-8') as file:
        api_data = json.load(file)

    base_url = api_data[0]["base_url"]
    api_key = api_data[0]["api_key"]
    model_cache = "../model_cache"

    chunk_agent = OpenaiAgent(base_url, api_key, model_name)
    kg_agent = OpenaiAgent(base_url, api_key, model_name)
    generate_agent = OpenaiAgent(base_url, api_key, model_name)

    embedding = HgEmbedding("nomic-ai/nomic-embed-text-v2-moe", model_cache)
    chunker = NaiveChunker("nomic-ai/nomic-embed-text-v2-moe", model_cache, max_token_length=750, context_sentence=0)
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe",cache_dir = model_cache)

    ans_rewrite_agent = OpenaiAgent(base_url, api_key, "gpt-4o-2024-11-20")
    ans_check_agent = OpenaiAgent(base_url, api_key, "gpt-4o-2024-11-20")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"./database/{test_name}/{{title}}/log_{timestamp}.log"





    with open(f"./dataset_cache/Hotpot/{dataset_name}", "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open(f"./dataset_cache/Hotpot/hotpot_dev_distractor_similar.json", "r", encoding="utf-8") as f:
        similar_data = json.load(f)
    with open(f"./dataset_cache/Hotpot/hotpot_dev_distractor_different.json", "r", encoding="utf-8") as f:
        different_data = json.load(f)

    if query_type=="origin":
        query = all_data[title_index]["question"]
        ans = all_data[title_index]["answer"]
    elif query_type=="similar":
        query = similar_data[title_index]["question"]
        ans = similar_data[title_index]["answer"]
    else:
        query = different_data[title_index]["question"]
        ans = different_data[title_index]["answer"]
    
    evidence = all_data[title_index]["supporting_facts"]
    context = all_data[title_index]["context"]




    need_load_data = os.path.exists(f"database/{test_name}/{title_index}")
    
    if not need_load_data:
        os.makedirs(f"database/{test_name}", exist_ok=True)
        os.makedirs(f"database/{test_name}/{title_index}", exist_ok=True)

    rag = ReMindRag(
        logger_level = 10,
        log_path= log_path.format(title=title_index),
        chunk_agent = chunk_agent, 
        kg_agent = kg_agent,
        generate_agent = generate_agent, 
        embedding = embedding,
        chunker = chunker,
        tokenizer = tokenizer,
        database_description = f"Database title: wiki",
        save_dir=f"database/{test_name}/{title_index}",
        edge_weight_coefficient=0.1,
        strong_connection_threshold=0.5
        
    )

    if not need_load_data:
        print(f"Load Data: {title_index}")
        for context_iter in context:
            context_str = context_iter[0] + "\n" + "\n".join(context_iter[1])
            rag.load_content(context_str, "en")
    else:
        print(f"{title_index} --- Data already loaded.")

    print(f"{title_index} Handle question")

    raw_response, chunks, edges = rag.generate_response(chat_history=[], user_input=query, do_update=False, force_do_rag=True, max_jumps=10)
    response = response_format.format(query = query, output = raw_response, chunks = str(chunks), edges = str(edges))

    ans_rewrite_input = ans_rewrite_prompt.format(question = query, og_output= response)
    rewrite_response = ans_rewrite_agent.generate_response("", [{"role":"user","content":ans_rewrite_input}])

    ans_check_input = ans_checck_prompt.format(question= query, reference_answer=ans, generated_output=rewrite_response)
    ans_check_response = ans_check_agent.generate_response("", [{"role":"user","content":ans_check_input}])

    all_inputs = {
        "query": query,
        "response": response,
        "rewrite_response":rewrite_response,
        "real_ans": ans,
        "evidence": evidence,
        "ans_check_input": ans_check_input,
        "check_response": ans_check_response
    }

    total_num += 1

    if ans_check_response == "True":
        print(f"{title_index} Get Right Ans")
        right_num += 1
    elif ans_check_response == "False":
        print(f"{title_index} Get Wrong Ans")
        pass
    else:
        print(f"Ans Check Output Error: {ans_check_response}")


    with open(f"database/{test_name}/{title_index}/input.json", "w", encoding="utf-8") as f:
        json.dump(all_inputs, f, ensure_ascii=False, indent=4)
    

    return {
        "title": title_index,
        "correct": ans_check_response
    }

if __name__ == "__main__":
    main()