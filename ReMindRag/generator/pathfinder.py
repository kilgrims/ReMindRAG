from ..llms import AgentBase
from ..database import ChromaDBManager
from .prompts import chunk_summary_prompt, judge_sufficient_information_prompt, find_next_node_prompt, reward_or_punishment_prompt, select_another_node_prompt


from ..utils.decorators import retry_json_parsing ,check_keys, unpack_cot_ans
from ..utils.logger import setup_logger
from typing import List, Dict
import logging



class PathFinder():
    def __init__(self, agent:AgentBase, database:ChromaDBManager, chunk_summary_threshold, logger_level, log_path):
        self.max_retries = 3

        self.agent = agent
        self.database = database
        self.chunk_summary_threshold = chunk_summary_threshold

        self.entity = []
        self.chunk = []
        self.edge = []
        self.path = []
        self.c_node_list = []
        self.out_dgree = {}
        self.summary = {}

        self.logger = setup_logger("PathFinder", logger_level, log_path)

        self.update_dict = {}
        self.confirm_paths = []
        self.viewed_nodes = []
        

    def get_query_ans(self, query, do_update, search_keys, max_jumps):
        self.logger.info("Start PathFinder Process...")
        self.entity = []
        self.chunk = []
        self.edge = []
        self.path = []
        self.c_node_list = []
        self.out_dgree = {}
        self.summary = {}

        self.logger.info(f"Get Query: {query}")
        self.query = query
        self.database_query = query


        query_embedding = self.database.embedding.sentence_embedding(query)
        results = self.database.entity_collection.query(
            query_embeddings=[query_embedding],
            n_results=search_keys
        )
        if not results["ids"][0]:
            self.logger.error(f"No Data Found. The data may not be loaded into the database.")
            raise RuntimeError("No Data Found. The data may not be loaded into the database.")

        self.entity = results["ids"][0]
        self.logger.debug(f"Initial Entity: {self.entity}")
        if len(self.entity) < search_keys:
            self.logger.warning(f"Can't found enough entities, only found ({len(self.entity)}/{search_keys})")
            search_keys = len(self.entity)

        for entity_iter in range(search_keys):
            origin_entities, origin_chunks, origin_edges = self.database.quick_query(self.database_query, self.entity[entity_iter])

            for edge in origin_edges:
                if edge["type"] == "relation":
                    if edge["to"] in self.entity:
                        self.logger.debug(f"Edge {edge['id']} May Form a Loop, Deleted.")
                        continue
                    self.edge.append(f"from entity:{edge['from']} to entity:{edge['to']} with relations: {edge['documents']}")
                    self.path.append({"from":edge['from'], "to":edge['to'], "id":edge['id'], "documents":edge['documents'],"type":"relation"})
                else:
                    if edge["to"] in self.chunk:
                        self.logger.debug(f"Connection {edge['id']} Already Exist, Deleted.")
                        continue
                    self.edge.append(f"from entity:{edge['from']} to chunk:{edge['to']}")
                    self.path.append({"from":edge['from'], "to":edge['to'], "id":edge['id'],"type":"connection"})

            for origin_entity in origin_entities:
                if origin_entity not in self.entity:
                    self.entity.append(origin_entity)

            for origin_chunk in origin_chunks:
                if origin_chunk not in self.chunk:
                    self.chunk.append(origin_chunk)
            

        for entity_iter in range(search_keys):
            node_id = self.entity[entity_iter]
            if node_id.startswith("anchor"):
                chunk_id = node_id.split("-")[1]
                if chunk_id not in self.chunk:
                    self.chunk.append(chunk_id)
                    self.edge.append(f"from entity:{node_id} to chunk:{chunk_id}")
                    self.path.append({"from":node_id, "to":chunk_id, "id":f"{node_id}_{chunk_id}", "type":"connection"})

        
        for entity_iter in self.entity:
            self.out_dgree[entity_iter], _ =self.get_out_dgree(entity_iter)
        
        self.logger.info(f"Quick Search Results --- entity:{len(self.entity)}, chunk:{len(self.chunk)}, edge:{len(self.edge)}.")
        self.logger.debug(f"Quick Search Entity List:\n{self.entity}")
        self.logger.debug(f"Quick Search Chunk List:\n{self.chunk}")
        self.logger.debug(f"Quick Search Edge List:\n{self.edge}")
        if self.chunk:
            for chunk_iter in self.chunk:
                self.summary[f"chunk:{chunk_iter}"] = self.get_chunk_summary(chunk_iter)
        self.logger.debug(f"Chunk Summary:\n{self.summary}")

        enough_str = self.judge_sufficient_information()
        if enough_str  == "yes":
            enough = True
            self.logger.info(f"Data Enough? {enough}")
        else:
            enough = False
            self.logger.info(f"Data Enough? {enough}")
            self.logger.info(f"Start Search At Entity Node:{self.entity[0]}.")
        c_node = self.entity[0]
        self.c_node_list.append(c_node)

        jumps = 0
        
        while not enough:
            self.logger.debug(f"Out Dgree: {self.out_dgree}")
            processed_relations = []
            processed_path = []
            processed_connection = []
            connection_chunk_id = ""
            if c_node:
                self.logger.info(f"======= Current Entity: {c_node} === Jumps: {jumps} ===========================================================")
                self.logger.debug(f"Entity Now:{self.entity}")
                self.logger.debug(f"Chunks Now:{self.chunk}")
                relations, connection = self.database.get_entity_edges(query, c_node)
                self.logger.trace(f"Relations:\n{relations}")
                self.logger.trace(f"Connections:\n{connection}")
                self.logger.info(f"Current Node Edge --- Relations:{len(relations)}, Connections:{len(connection)==6}.")

                for relation in relations:
                    processed_relation_iter = f"from entity:{relation['from']} to entity:{relation['to']} with relations: {relation['documents']}"

                    if relation["to"] not in self.entity:
                        self.logger.trace(f"Relation: {processed_relation_iter}")
                        processed_relations.append(processed_relation_iter)
                        processed_path.append({"from":relation['from'], "to":relation['to'], "id":relation['id'], "documents":relation['documents'], "type":"relation"})
                    else:
                        self.logger.debug(f"Entity {relation['to']} Already in Entity Set.")
                
                if connection:
                    if connection["to"] not in self.chunk:
                        processed_connection = f"from entity:{connection['from']} to chunk:{connection['to']}"
                        processed_path.append({"from":connection['from'], "to":connection['to'], "id":connection['id'], "type":"connection"})
                        connection_chunk_id = connection["to"]
                        self.logger.trace(f"Connection: {processed_connection}")
            else:
                self.logger.info(f"======= Current Node: In Chunk Handle Process ================================================================")
                
            anchor_chunk_titles = self.get_anchor_chunk_title(processed_path)
            self.logger.debug(f"Find Next Node... With Data Size --- edges:{len(str(self.edge+processed_relations))+len(processed_connection)}, chunk summary:{len(str(self.summary))}, anchor chunk titles:{len(str(anchor_chunk_titles))}")
            
            if connection_chunk_id:
                new_node = f"chunk:{connection_chunk_id}"
                new_node_type = "chunk"
                new_node_id = connection_chunk_id
                self.edge.append(f"from entity:{c_node} to chunk:{connection_chunk_id}")
                self.path.append({"from":c_node, "to":connection_chunk_id, "id":f"{c_node}_{connection_chunk_id}", "type":"connection"})

            elif sum(self.out_dgree.values()) == 0:
                self.logger.error(f"No More Nodes Can Be Found,")
                # do_update = False
                break
            else:
                find_chunk_already_in = []
                for attempt in range(self.max_retries):
                    try:
                        temp_ans = False
                        new_node = self.find_next_node(c_node, processed_relations, processed_connection, anchor_chunk_titles, find_chunk_already_in)
                        if not len(new_node.split(":")) == 2:
                            self.logger.error(f"New Node Format Error. Retrying...({attempt+1}/{self.max_retries})")
                            continue
                        new_node_type = new_node.split(":")[0]
                        new_node_id = new_node.split(":")[1]
                        if new_node_type == "chunk" or new_node_type == "entity":
                            for path_iter in processed_path:
                                if path_iter["type"] == "connection" and path_iter["to"] == new_node_id and new_node_type == "chunk":
                                    self.edge.append(f"from entity:{path_iter['from']} to chunk:{path_iter['to']}")
                                    self.path.append({"from":path_iter['from'], "to":path_iter['to'], "id":path_iter['id'], "type":"connection"})
                                    temp_ans= True
                                    break
                                if path_iter["type"] == "relation" and path_iter["to"] == new_node_id and new_node_type == "entity":
                                    self.edge.append(f"from entity:{path_iter['from']} to entity:{path_iter['to']} with relations: {path_iter['documents']}")
                                    self.path.append({"from":path_iter['from'], "to":path_iter['to'], "id":path_iter['id'], "type":"relation"})
                                    temp_ans = True
                                    break
                        if temp_ans:
                            break
                        for entity_check_iter in self.entity:
                            if new_node_type == "entity" and new_node_id == entity_check_iter:
                                temp_ans = True
                                break
                        if temp_ans:
                            break
                        if new_node_type=="chunk":
                            anchor_temp_id = f"anchor-{new_node_id}"
                            # if anchor_temp_id in self.entity:
                            if (anchor_temp_id in self.entity) and (new_node_id not in self.chunk):
                                self.edge.append(f"from entity:{anchor_temp_id} to chunk:{new_node_id}")
                                self.path.append({"from":anchor_temp_id, "to":new_node_id, "id":f"{anchor_temp_id}_{new_node_id}", "type":"connection"})
                                temp_ans = True
                                break

                            if new_node_id in self.chunk:
                                find_chunk_already_in = [{"role":"assistant","content":new_node},{"role":"user","content":select_another_node_prompt.format(node = new_node, chunks = self.chunk)}]
                        if temp_ans:
                            break
                        self.logger.error(f"No node was identified during the search for a new node. Retrying...({attempt+1}/{self.max_retries})")
                    except Exception as e:
                        raise RuntimeError(f"No node was identified in Knowledge Graph during the search for a new node.")
                    
            
                if not temp_ans:
                    self.logger.error("Failed to find a valid new node after maximum retries.")
                    do_update = False
                    break
            
            self.logger.info(f"Select Next Node: {new_node}")
            self.c_node_list.append(c_node)
            if new_node_type == "chunk":
                c_node = ""
                jumps += 1
                chunk_id = new_node_id
                self.chunk.append(chunk_id)
                self.logger.info(f"Step into Chunk: {chunk_id}")
                self.logger.info(f"Update Chunk Summary with Chunk:{chunk_id}.")
                self.summary.update({f"chunk:{chunk_id}":self.get_chunk_summary(chunk_id)})
                self.logger.debug(f"New Chunk Summary:\n{self.summary}")
            else:
                entity_id = new_node_id
                if entity_id in self.entity:
                    self.logger.info(f"Go Back to Entity: {entity_id}")
                else:
                    jumps += 1
                    self.logger.info(f"Step into Entity: {entity_id}")
                    self.entity.append(entity_id)
                    self.out_dgree[entity_id], has_out_entity=self.get_out_dgree(entity_id)
                    for has_out_entity_iter in has_out_entity:
                        self.out_dgree[has_out_entity_iter] -= 1
                c_node = entity_id
            
            enough_str = self.judge_sufficient_information()
            if enough_str  == "yes":
                enough = True
            else:
                enough = False
            self.logger.info(f"Data Enough? {enough}")

            
            if jumps >= max_jumps:
                self.logger.error(f"Reach the Max Jumps: {max_jumps}")
                # do_update = False
                break
            
            self.logger.info("=============================================================================================================")
        
        

        self.logger.info(f"Stop Search --- Jumps: {jumps}")
        if jumps == 0:
            self.logger.info(f"No LLM-Guided Traversal, Skip Update.")
            do_update = False
        self.logger.debug(f"Start Node: {self.entity[:search_keys]}")
        self.logger.debug(f"Entities : {self.entity}")
        self.logger.debug(f"Chunks : {self.chunk}")
        self.logger.debug(f"Paths:\n{self.path}")
        if do_update:
            self.logger.info("Update Knowledge Graph.")

            final_confirm_paths = []
            punish_paths = []
            self.update_dict = self.get_update_relation_and_chunk()
            str_chunk_list = []
            for update_chunk_iter in self.update_dict["chunks"]:
                str_chunk_list.append(str(update_chunk_iter))
            self.update_dict["chunks"] = str_chunk_list

            self.logger.debug(f"Update Dict: {self.update_dict}")
            
            for entity_iter in range(search_keys):
                self.viewed_nodes = []
                self.confirm_paths = []
                if not self.get_update_dfs(self.entity[entity_iter]):
                    self.logger.warning(f"Update: Can't Find Available Path (Start with {self.entity[entity_iter]}).")
                else:
                    final_confirm_paths = final_confirm_paths + self.confirm_paths
                    
            for path_iter in self.path:
                path_iter_tuple = (path_iter["type"],path_iter["id"])
                if path_iter_tuple not in final_confirm_paths:
                    punish_paths.append(path_iter_tuple)

            self.logger.info(f"Enhance {len(final_confirm_paths)} Edges, Punish {len(punish_paths)}.")
            
            self.database.enhance_edge_weight(self.database_query, final_confirm_paths)
            self.logger.debug(f"Enhance Edges:\n{final_confirm_paths}")
            self.database.punish_edge_weight(self.database_query, punish_paths)
            self.logger.debug(f"Punish Edges:\n{punish_paths}")
        else:
            self.logger.info("Skip Update.")
                
        return self.summary, self.edge



    def get_chunk_summary(self, chunk_id):
        chunk_data = self.database.chunk_collection.get(ids = [chunk_id], include=['documents','metadatas'])
        chunk_tokens = chunk_data["metadatas"][0]['tokens']
        chunk_document = chunk_data["documents"][0]
        if chunk_tokens < self.chunk_summary_threshold:
            return chunk_document
        

        input_msg = chunk_summary_prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            edge_list = str(self.edge),
            chunk_summary = self.summary,
            chunk_document = chunk_document
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}])
        return response

    @unpack_cot_ans
    def judge_sufficient_information(self, error_chat_history = None):
        input_msg = judge_sufficient_information_prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            chunk_list = str(self.chunk),
            edge_list = str(self.edge),
            chunk_summary = self.summary
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}]+ (error_chat_history or []))
        self.logger.debug(f"Function judge_sufficient_information Output:\n{response}")
        return response

    @unpack_cot_ans
    def find_next_node(self, c_node, relation, connection, anchor_chunk_titles, find_chunk_already_in, error_chat_history = None):
        input_msg = find_next_node_prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            chunk_list = str(self.chunk),
            edge_list = str(self.edge),
            anchor_chunk_titles = str(anchor_chunk_titles),
            chunk_summary = self.summary,
            c_node = c_node,
            c_node_list = str(self.c_node_list),
            relation_cnode = str(relation),
            connection_cnode = str(connection)
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + find_chunk_already_in + (error_chat_history or []))
        self.logger.debug(f"Function find_next_node Output:\n{response}")
        return response
    
    @check_keys("edges","chunks")
    @unpack_cot_ans
    def get_update_relation_and_chunk(self, error_chat_history = None):
        input_msg = reward_or_punishment_prompt.format(
            query = self.query,
            entity_list = str(self.entity),
            chunk_list = str(self.chunk),
            edge_list = str(self.path),
            chunk_summary = self.summary,
        )
        response = self.agent.generate_response("", [{"role":"user","content":input_msg}] + (error_chat_history or []))
        self.logger.debug(f"Function get_update_relation_and_chunk Output:\n{response}")
        return response
    
    def get_update_dfs(self, node_id):
        if node_id in self.viewed_nodes:
            return False
        ans = False
        for node_path in self.path:
            if node_path["from"] == node_id:
                if node_path["to"] in self.viewed_nodes:
                    continue
                if node_path["type"] == "relation":
                    temp_dfs_ans = self.get_update_dfs(node_path["to"])
                    if (not ans) and (node_path["id"] in self.update_dict["edges"] or temp_dfs_ans):
                        self.confirm_paths.append(("relation",node_path["id"]))
                        ans = True
                else:
                    if (not ans) and node_path["to"] in self.update_dict["chunks"]:
                        self.confirm_paths.append(("connection",node_path["id"]))
                        ans = True

                    
        self.viewed_nodes.append(node_id)
        return ans



    def get_anchor_chunk_title(self, paths):
        anchor_chunk_title = []
        for entity_iter in self.entity:
            if entity_iter.startswith("anchor"):
                chunk_title = self.database.entity_collection.get(ids=[entity_iter],include=["documents"])["documents"][0]
                anchor_chunk_title.append(f"{entity_iter} connect to chunk which have document:{chunk_title}")

        for path in paths:
            if path["type"] == "relation":
                if path["to"].startswith("anchor"):
                    chunk_title = self.database.entity_collection.get(ids=[path["to"]],include=["documents"])["documents"][0]
                    anchor_chunk_title.append(f"{path['to']} connect to chunk which have document:{chunk_title}")
            else:
                if path["from"].startswith("anchor"):
                    chunk_title = self.database.entity_collection.get(ids=[path["from"]],include=["documents"])["documents"][0]
                    anchor_chunk_title.append(f"{path['from']} connect to chunk:{path['to']} which have document:{chunk_title}")
        
        return anchor_chunk_title
    

    def get_out_dgree(self, entity_id):
        out_dgree = 0
        has_out_entity = []
        relations = self.database.relation_collection.get(
            where={
                "$or": [
                    {"subject_entity_id": entity_id},
                    {"object_entity_id": entity_id}
                ]
            },
            include=['metadatas']
        )

        for relation_iter in range(len(relations["ids"])):
            
            another_node = ""
            if relations["metadatas"][relation_iter]["subject_entity_id"] == entity_id:
                another_node = relations["metadatas"][relation_iter]["object_entity_id"]
            else:
                another_node = relations["metadatas"][relation_iter]["subject_entity_id"]
            
            if another_node not in self.entity:
                out_dgree += 1
            else:
                has_out_entity.append(another_node)
        connection = self.database.connection_collection.get(where={"entity_id":entity_id}, include=['metadatas', 'documents', 'embeddings'])
        if connection["ids"]:
            if str(connection["metadatas"][0]["chunk_id"]) not in self.chunk:
                out_dgree += 1

        return out_dgree, has_out_entity
