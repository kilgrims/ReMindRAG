from ..llms import AgentBase
from ..database import ChromaDBManager
from .prompts import analyze_input_is_question_or_not_prompt, rewrite_prompt, generate_rag_ans_prompt, split_question_prompt, generate_final_ans_prompt, split_question_rewrite_prompt
from .pathfinder import PathFinder

from ..utils.decorators import retry_json_parsing ,check_keys, unpack_cot_ans
from ..utils.logger import setup_logger
from typing import List, Dict
import logging
import json
import os

import concurrent.futures

class PreProcessing():
    def __init__(self, agent:AgentBase, kg_agent:AgentBase, database:ChromaDBManager, daatabase_description:str, chunk_summary_threshold, logger_level, save_dir, log_path):
        self.agent = agent
        self.database = database
        self.database_description = daatabase_description
        self.max_retries = 3
        self.path_finder = PathFinder(kg_agent, database, chunk_summary_threshold, logger_level, log_path)
        self.save_dir = save_dir
        self.logger = setup_logger("PreProcessing", logger_level, log_path)

        if not os.path.exists(f"{self.save_dir}/split_question_cache.json"):
            with open(f"{self.save_dir}/split_question_cache.json", 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
            self.logger.info(f"Created New Cache : {f'{self.save_dir}/split_question_cache.json'}")

    def query_main(self, system_prompt, chat_history, user_input, search_key_nums, max_jumps, max_split_question_num, force_do_rag, do_update):
        self.logger.info("---------------------------------------------------------------------------------------------------------")
        self.logger.info("Start Query Process...")

        if not force_do_rag:
            need_rag_str = self.check_need_rag(chat_history, user_input)
            if need_rag_str == "yes":
                self.logger.info("Perform a RAG search.")
                need_rag = True
            else:
                need_rag = False
        else:
            self.logger.info("Force do Rag Seach")
            need_rag = True
        response = ""
        if need_rag:
            chat_history_str = self.change_chat_history_to_str(chat_history)
            rewritten_querys = self.get_spilt_question(chat_history_str, user_input, max_split_question_num)

            rewritten_query_and_ans = {}
            final_chunk_summary = []
            final_edges = []

            if len(rewritten_querys) == 1:
                final_chunk_summary, final_edges = self.path_finder.get_query_ans(rewritten_querys[0], do_update, search_key_nums, max_jumps)
                response = self.generate_temp_response(chat_history_str, system_prompt, rewritten_querys[0], final_chunk_summary, final_edges)
                final_chunk_summary = [final_chunk_summary]
            else:
                for rewritten_query in rewritten_querys:
                    chunk_summary, edges = self.path_finder.get_query_ans(rewritten_query, do_update, search_key_nums, max_jumps)
                    final_chunk_summary.append(chunk_summary)
                    final_edges = final_edges + edges
                    temp_response = self.generate_temp_response(chat_history_str, system_prompt, rewritten_query, chunk_summary, edges)
                    rewritten_query_and_ans[rewritten_query] = temp_response
                    self.logger.info(f"Splited Question: {rewritten_query}\n Ans:{temp_response}")
                response = self.generate_final_response(chat_history_str, user_input, rewritten_query_and_ans)
            
            self.logger.info(f"Origin Query: {user_input}")
            self.logger.info(f"Response:\n{response}")
            self.logger.info("End PreProcessing Process.")
            return response, final_chunk_summary, final_edges
            
        else:
            self.logger.debug("Skip the RAG search, generate directly.")
            response = self.agent.generate_response(system_prompt, chat_history+[{"role":"user","content":user_input}])
            return response, [], []

        

    def change_chat_history_to_str(self, chat_history):
        chat_history_str = ""
        for history_iter in chat_history:
            chat_history = chat_history + f"{history_iter['role']}: {history_iter['content']}" + "\n"
        return chat_history_str
    
    # def process_query(self, rewritten_query, chat_history_str, system_prompt, do_update, search_key_nums, max_jumps):
    #     chunk_summary, edges = self.path_finder.get_query_ans(rewritten_query, do_update, search_key_nums, max_jumps)
    #     temp_response = self.generate_temp_response(chat_history_str, system_prompt, rewritten_query, chunk_summary, edges)
    #     self.logger.info(f"Splited Question: {rewritten_query}\n Ans:{temp_response}")
    #     return rewritten_query, chunk_summary, edges, temp_response

    
    @retry_json_parsing
    @unpack_cot_ans
    def splite_question(self, chat_history_str, user_input, max_split_question_num, temp_error_chat_history, error_chat_history = None):
        input_msg = split_question_prompt.format(chat_history = chat_history_str, user_input = user_input, max_split_question_num = max_split_question_num)
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + temp_error_chat_history + (error_chat_history or []))
        self.logger.debug(f"Function: splite_question Output:\n{response}")
        return response
    
    def get_spilt_question(self, chat_history_str, user_input, max_split_question_num):
        if max_split_question_num == 1:
            self.logger.info(f"Origin Query: {user_input}")
            # self.logger.info(f"Splited Question: {user_input}")
            return [user_input]

        self.logger.info(f"Origin Query: {user_input}")
        

        temp_error_chat_history = []
        for attempt in range(self.max_retries):
            try:
                rewritten_querys = self.splite_question(chat_history_str, user_input, max_split_question_num, temp_error_chat_history)
            except:
                rewritten_querys = [user_input]
            if len(rewritten_querys) <= max_split_question_num and len(rewritten_querys) > 0:
                break
            else:
                error_info = f"The number of queries after rewriting is incorrect --- ({len(rewritten_querys)}/{max_split_question_num})"
                temp_error_chat_history= temp_error_chat_history + [{"role":"assistant","content":str(rewritten_querys)},{"role":"user","content":split_question_rewrite_prompt.format(error = error_info)}]
                self.logger.warning(error_info)
                if attempt == (self.max_retries-1):
                    raise RuntimeError("The number of queries after rewriting is incorrect. Reached Max Tries")
        
        self.logger.info(f"Splited Question: {rewritten_querys}")
        return rewritten_querys
        

    @unpack_cot_ans
    def generate_final_response(self, chat_history_str, origin_query, rewritten_query_and_ans, error_chat_history = None):
        input_msg = generate_final_ans_prompt.format(chat_history_str = chat_history_str, origin_query = origin_query, rewritten_query_and_ans = rewritten_query_and_ans)
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + (error_chat_history or []))
        return response

    @unpack_cot_ans
    def generate_temp_response(self, chat_history_str, system_prompt, rewritten_query, chunk_summary, edges, error_chat_history = None):
        chunk_summary_str = ""
        if chunk_summary:
            for key, item in chunk_summary.items():
                # chunk_summary_str = chunk_summary_str + item + "\n"
                chunk_summary_str = chunk_summary_str + key + ":" + item + "\n"
        self.logger.debug(f"Get Rag Chunk Summary:\n{chunk_summary_str}")
        if edges:
            edges_str = str(edges)
        else:
            edges_str = ""
        self.logger.debug(f"Get Rag Edges:\n{edges_str}")

        input_msg = generate_rag_ans_prompt.format(chat_history = chat_history_str, query = rewritten_query, rag_summary = chunk_summary_str, edges=edges_str)
        response = self.agent.generate_response(system_prompt, [{"role":"user","content":input_msg}] + (error_chat_history or []))
        return response

    @unpack_cot_ans
    def check_need_rag(self, chat_history, user_input, error_chat_history = None):
        chat_history_str = self.change_chat_history_to_str(chat_history)
        input_msg = analyze_input_is_question_or_not_prompt.format(chat_history = chat_history_str, user_input = user_input,database_description = self.database_description)
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + (error_chat_history or []))
        return response
        
    @unpack_cot_ans
    def query_rewrite(self, chat_history, user_input, error_chat_history = None):
        chat_history_str = self.change_chat_history_to_str(chat_history)
        input_msg = rewrite_prompt.format(chat_history = chat_history_str, user_input = user_input)
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + (error_chat_history or []))
        return response