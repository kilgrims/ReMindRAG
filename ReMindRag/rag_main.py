from .llms import AgentBase
from .embeddings import EmbeddingBase
from .chunking import ChunkerBase
from .database import ChromaDBManager
from .kg import KnowledgeGraph
from .generator import PreProcessing
from .utils.logger import setup_logger, trace

from typing import List,Dict
from transformers import AutoTokenizer
import logging


class ReMindRag:
    def __init__(self, 
                logger_level:int,
                chunk_agent:AgentBase, 
                kg_agent:AgentBase,
                generate_agent:AgentBase, 
                embedding:EmbeddingBase, 
                chunker:ChunkerBase, 
                tokenizer:AutoTokenizer,

                database_description:str,
                
                chunk_summary_threshold = 350,
                synonym_threshold = 0.7, 
                edge_weight_coefficient = 0.1, 
                strong_connection_threshold = 0.6,
                
                save_dir:str = './Rag_Cache', 
                log_path = None
                ):

        logging.addLevelName(5, "TRACE")
        logging.Logger.trace = trace
        self.logger = setup_logger("DynamicRag", logger_level, log_path)

        print(f"Logger Level: {self.logger.getEffectiveLevel()}.")
        self.logger.info("Start Initail DynamicRag.")

        self.save_dir = save_dir
        self.database_pth  = save_dir + "/chroma_data"
        self.model_cache = save_dir + "/model_cache"

        self.chunk_agent = chunk_agent
        self.kg_agent = kg_agent
        self.generate_agent = generate_agent
        self.embedding = embedding
        self.chunker = chunker

        
        self.database = ChromaDBManager(
            logger_level = logger_level,
            chunk_agent = chunk_agent,
            embedding = embedding,
            chunker = chunker,
            synonym_threshold = synonym_threshold,
            edge_weight_alpha = edge_weight_coefficient,
            strong_connection_threshold = strong_connection_threshold,
            chromadp_pth = self.database_pth,
            tokenizer = tokenizer,
            log_path = log_path
            )
        
        self.kg = KnowledgeGraph(
            chroma_manager = self.database,
            logger_level = logger_level,
            log_path = log_path
        )

        self.preprocess = PreProcessing(
            agent=generate_agent, 
            kg_agent=kg_agent,
            database=self.database, 
            daatabase_description=database_description,
            chunk_summary_threshold = chunk_summary_threshold,
            logger_level=logger_level,
            save_dir = save_dir,
            log_path=log_path
        )

        self.logger.info("Initialize DynamicRag Successfully.")
        

    def set_database_description(self,database_description):
        self.preprocess.database_description = database_description
    
    def load_content(self, content, language = "en"):
        self.logger.info(f"Load Content:{content[:10]}...")
        self.database.add_content(content, language)
    
    def load_file(self, file_pth, language = "en", encoding="utf-8"):
        self.logger.info(f"Load File: {file_pth}.")
        self.database.add_file_data(file_pth=file_pth, language=language, encoding=encoding)

    def load_folder(self, folder_pth, language = "en", encoding="utf-8"):
        self.logger.info(f"Load Folder: {folder_pth}.")
        self.database.add_folder_data(folder_file_pth=folder_pth,language=language,encoding=encoding)
    
    def refresh_kg(self):
        self.kg.load_graph()
    
    def export_kg_as_pyvis(self, save_pth = None):
        if not save_pth:
            save_pth = self.save_dir+"/graph.html"
        self.kg.save_as_pyvis(save_pth)
        self.logger.info(f"Save knowledge graph in {save_pth}.")

    

    def generate_response(
            self,
            user_input:str,
            chat_history:List[Dict[str,str]] = [],
            system_prompt:str = None,
            search_key_nums = 2,
            max_jumps = 10,
            force_do_rag=False, 
            do_update:bool = True):
        
        if not system_prompt:
            system_prompt = """
            You will now act as a database Q&A bot. 
            You'll receive information from the backend database and answer questions accordingly. 
            Please respond using the most concise language possible, without including any extra information.
            """
        response, chunks, edges = self.preprocess.query_main(
            system_prompt=system_prompt, 
            chat_history=chat_history, 
            user_input=user_input, 
            search_key_nums=search_key_nums,
            max_jumps=max_jumps,
            force_do_rag=force_do_rag,
            do_update=do_update)
        return response, chunks, edges