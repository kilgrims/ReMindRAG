from ..llms import AgentBase
from ..utils.decorators import check_keys, retry_json_parsing
from ..chunking import ChunkerBase
from .prompts import entity_extract_prompt, relation_extract_prompt, chunk_title_get_prompt, relation_num_error_rewrite_prompt

import json
import os
from typing import List, Dict

from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx import Document
import logging

@check_keys()
def generate_entity_response(agent:AgentBase, chunk_text:str, error_chat_history = None):
    response = agent.generate_response(entity_extract_prompt, [{"role": "user", "content": chunk_text}] + (error_chat_history or []))
    return response


@check_keys()
def generate_relation_response(agent:AgentBase, chunk_text:str, entity_list, temp_error_chat_history, error_chat_history = None):
    response = agent.generate_response(relation_extract_prompt.format(entity_list=entity_list), [{"role": "user", "content": chunk_text}] + temp_error_chat_history + (error_chat_history or []))
    return response

def generate_chunk_title(agent:AgentBase , chunk_text:str):
    response = agent.generate_response(chunk_title_get_prompt, [{"role": "user", "content": chunk_text}])
    return response



def handle_content(logger, content:str, agent:AgentBase, chunker:ChunkerBase, language:str) -> List[Dict[str,str]]:
    max_retries = 3
    logger.info("Chunking...")
    chunks = chunker.chunk_text(content, language)
    logger.info(f"Get {len(chunks)} Chunks.")
    extracted_text = []

    for chunk_num, chunk in enumerate(chunks):
        logger.info(f"Do Infomation Extraction in Chunk {chunk_num}/{len(chunks)}")
        # print(f"================================\nchunk now:\n{chunk}")
        entity_list = generate_entity_response(agent, chunk)
        temp_error_chat_history = []
        for i in range(max_retries):
            relation_check = True
            relation_list = generate_relation_response(agent, chunk, entity_list, temp_error_chat_history)
            # print(f"================================\nresponse:\n{chunk_json_data}")
            temp_error_chat_history.append({"role":"assistant","content":str(relation_list)})
            for relation_iter in relation_list:
                if len(relation_iter) != 3:
                    temp_error_chat_history.append({"role":"user","content":relation_num_error_rewrite_prompt.format(relation = str(relation_iter))})
                    relation_check = False
                    break
            if relation_check:
                break

        chunk_title = generate_chunk_title(agent, chunk)
        # chunk_title = chunk
        extracted_text_iter = {}
        extracted_text_iter["chunk"] = {"title":chunk_title, "content":chunk}
        extracted_text_iter["entity"] = entity_list
        extracted_text_iter["relation"] =relation_list
        extracted_text.append(extracted_text_iter)
    
    logger.info("Finish Infomation Extraction.")
    return extracted_text







def handle_txt_file(file_pth:str, encoding):
    with open(file_pth,"r",encoding=encoding) as f:
        file_data = f.read()
    return file_data

def handle_docx_file(file_pth:str, encoding) -> List[Dict[str,str]]:

    def iter_block_items(parent):
        if isinstance(parent, DocxDocument):
            parent_elm = parent.element.body
        else:
            parent_elm = parent._tc
        for child in parent_elm.iterchildren():
            # 判断标签是否为段落
            if child.tag == qn('w:p'):
                yield Paragraph(child, parent)
            # 判断标签是否为表格
            elif child.tag == qn('w:tbl'):
                yield Table(child, parent)

    doc = Document(file_pth)
    output_text = ""
    # 遍历所有块级元素，确保输出顺序与文档一致
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            output_text += block.text + "\n"
        elif isinstance(block, Table):
            # 处理表格，将每一行中各单元格用制表符分隔
            for row in block.rows:
                row_text = "\t".join(cell.text for cell in row.cells)
                output_text += row_text + "\n"

    return output_text





def handle_file(logger, agent: AgentBase, chunker: ChunkerBase, file_pth: str, language:str, encoding:str):
    if not os.path.exists(file_pth):
        raise FileNotFoundError(f"Data file path does not exist: {file_pth}")
    
    data = []
        
    if os.path.isfile(file_pth):
        if file_pth.endswith(".txt"):
            data = handle_txt_file(file_pth, encoding)
        elif file_pth.endswith(".docx"):
            data = handle_docx_file(file_pth, encoding)
        else:
            print(f"Unsupported file types: {file_pth}")
    
    extracted_data = handle_content(logger, data, agent, chunker, language)

    if not extracted_data:
        print("No data found.")
    
    return extracted_data



def handle_file_folder(logger, agent: AgentBase, chunker: ChunkerBase, folder_pth: str, language:str, encoding:str):
    if not os.path.exists(folder_pth):
        raise FileNotFoundError(f"Data folder path does not exist: {folder_pth}")
    
    extracted_data = []
    for filename in os.listdir(folder_pth):
        file_path = os.path.join(folder_pth, filename)
        
        if os.path.isfile(file_path):
            if filename.endswith(".txt"):
                data = handle_txt_file(file_path, encoding)
            else:
                print(f"Unsupported file types: {filename}")
        
        extracted_data = extracted_data + handle_content(logger, data, agent, chunker, language)

    if not extracted_data:
        print("No data found.")

    return extracted_data

