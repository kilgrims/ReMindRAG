from ..database import ChromaDBManager
from ..utils.logger import setup_logger

import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from typing import  Optional, Tuple
import copy
import numpy as np
import logging

class KnowledgeGraph:
    def __init__(self, chroma_manager: ChromaDBManager, logger_level, log_path):
        self.chroma_manager = chroma_manager
        self.graph = nx.DiGraph()

        self.logger = setup_logger("KG", logger_level, log_path)
        
    def load_graph(self):
        self.graph.clear()
        
        entities = self.chroma_manager.get_all_entities()
        for entity_id, entity_data in entities.items():
            self.graph.add_node(entity_id, type='entity', label=entity_id)
        
        relations = self.chroma_manager.get_all_relations()
        for relation_id, relation_data in relations.items():
            subject_id = relation_data['subject_entity_id']
            object_id = relation_data['object_entity_id']
            relation_text = relation_data['id']
            
            if subject_id in self.graph and object_id in self.graph:
                self.graph.add_edge(subject_id, object_id, type='relation', label=relation_text)
        
        chunks = self.chroma_manager.chunk_collection.get()
        for chunk_id, chunk_data in zip(chunks["ids"], chunks["documents"]):
            self.graph.add_node(chunk_id, type='chunk', label=f"Chunk {chunk_id}")
        
        connections = self.chroma_manager.connection_collection.get()
        for connection_id, connection_data in zip(connections["ids"], connections["metadatas"]):
            entity_id = connection_data["entity_id"]
            chunk_id = connection_data["chunk_id"]
            
            
            if entity_id in self.graph and chunk_id in self.graph:
                self.graph.add_edge(entity_id, chunk_id, type='connection', label=connection_id)
        
        self.logger.info("Graph Loaded From Database Successfully.")
        return self.graph
    

    def save_as_pyvis(self, save_pth):
        net = Network(directed=False)
        temp_graph = copy.deepcopy(self.graph)

        chunk_x = -1000
        chunk_x_step = 200

        for node_id, attr in temp_graph.nodes(data=True):
            if 'type' in attr:
                if attr['type'] == 'entity':
                    temp_graph.nodes[node_id]['size'] = 25
                    if node_id.startswith("anchor-"):
                        id_int = int(node_id.split("-")[-1])
                        temp_graph.nodes[node_id]['color'] = '#D98A12'
                        temp_graph.nodes[node_id]['x'] = chunk_x + (id_int-1) * chunk_x_step
                        temp_graph.nodes[node_id]['y'] = -300
                        temp_graph.nodes[node_id]['fixed'] = True
                    else:
                        temp_graph.nodes[node_id]['color'] = '#4CAF50'
                elif attr['type'] == 'chunk':
                    temp_graph.nodes[node_id]['size'] = 20
                    temp_graph.nodes[node_id]['color'] = '#2196F3'
                    id_int = int(node_id)
                    temp_graph.nodes[node_id]['x'] = chunk_x + (id_int-1) * chunk_x_step
                    temp_graph.nodes[node_id]['y'] = -500
                    temp_graph.nodes[node_id]['fixed'] = True
                else:
                    temp_graph.nodes[node_id]['size'] = 15
        
        for u, v, attr in temp_graph.edges(data=True):
            temp_graph.edges[u, v]['label'] = ''
            temp_graph.edges[u, v]['width'] = 2
            
            if 'type' in attr:
                if attr['type'] == 'relation':
                    temp_graph.edges[u, v]['color'] = '#FF5722'
                elif attr['type'] == 'connection':
                    temp_graph.edges[u, v]['color'] = '#9E9E9E'
        
        net.from_nx(temp_graph)
        
        net.set_options("""
        const options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            },
            "nodes": {
                "font": {
                    "size": 12,
                    "face": "Tahoma"
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": false
                    }
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            }
        }
        """)
        
        net.width = "100%"
        net.height = "1200px"

        net.save_graph(save_pth)
        self.logger.info(f"Save Graph HTML File to {save_pth}")


    def save_as_pyvis_for_quick_query(self, save_pth, query, search_keys = 3):
        highlight_entities = []
        highlight_chunks = []
        highlight_relations =  []
        highlight_connections = []


        query_embedding = self.chroma_manager.embedding.sentence_embedding(query)
        results = self.chroma_manager.entity_collection.query(
            query_embeddings=[query_embedding],
            n_results=search_keys
        )
        if not results["ids"][0]:
            self.logger.error(f"No Data Found. The data may not be loaded into the database.")
            raise RuntimeError("No Data Found. The data may not be loaded into the database.")

        highlight_entities = results["ids"][0]
        self.logger.debug(f"Initial Entity: {highlight_entities}")
        if len(highlight_entities) < search_keys:
            self.logger.warning(f"Can't found enough entities, only found ({len(highlight_entities)}/{search_keys})")
            search_keys = len(highlight_entities)

        for entity_iter in range(search_keys):
            origin_entities, origin_chunks, origin_edges = self.chroma_manager.quick_query(query, highlight_entities[entity_iter])

            for edge in origin_edges:
                if edge["type"] == "relation":
                    if edge["to"] in highlight_entities:
                        self.logger.debug(f"Edge {edge['id']} May Form a Loop, Deleted.")
                        continue
                    highlight_relations.append(edge["id"])
                else:
                    if edge["to"] in highlight_chunks:
                        self.logger.debug(f"Connection {edge['id']} Already Exist, Deleted.")
                        continue
                    highlight_connections.append(["id"])

            for origin_entity in origin_entities:
                if origin_entity not in highlight_entities:
                    highlight_entities.append(origin_entity)

            for origin_chunk in origin_chunks:
                if origin_chunk not in highlight_chunks:
                    highlight_chunks.append(origin_chunk)


        net = Network(directed=False)
        temp_graph = copy.deepcopy(self.graph)

        chunk_x = -1000
        chunk_x_step = 200

        # Process nodes (entities and chunks)
        for node_id, attr in temp_graph.nodes(data=True):
            if 'type' in attr:
                # Default opacity for non-highlighted nodes
                opacity = 0.3
                
                if attr['type'] == 'entity':
                    temp_graph.nodes[node_id]['size'] = 25
                    
                    # Check if this entity should be highlighted
                    if node_id in highlight_entities:
                        opacity = 1.0
                    
                    if node_id.startswith("anchor-"):
                        id_int = int(node_id.split("-")[-1])
                        base_color = '#D98A12'
                        temp_graph.nodes[node_id]['x'] = chunk_x + (id_int-1) * chunk_x_step
                        temp_graph.nodes[node_id]['y'] = -300
                        temp_graph.nodes[node_id]['fixed'] = True
                    else:
                        base_color = '#4CAF50'
                    
                    # Apply opacity to color (convert hex to rgba)
                    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
                    temp_graph.nodes[node_id]['color'] = f'rgba({r}, {g}, {b}, {opacity})'
                    
                elif attr['type'] == 'chunk':
                    temp_graph.nodes[node_id]['size'] = 20
                    
                    # Check if this chunk should be highlighted
                    if node_id in highlight_chunks:
                        opacity = 1.0
                    
                    base_color = '#2196F3'
                    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
                    temp_graph.nodes[node_id]['color'] = f'rgba({r}, {g}, {b}, {opacity})'
                    
                    id_int = int(node_id)
                    temp_graph.nodes[node_id]['x'] = chunk_x + (id_int-1) * chunk_x_step
                    temp_graph.nodes[node_id]['y'] = -500
                    temp_graph.nodes[node_id]['fixed'] = True
                else:
                    temp_graph.nodes[node_id]['size'] = 15
        
        # Process edges (relations and connections)
        for u, v, attr in temp_graph.edges(data=True):
            # Default opacity for non-highlighted edges
            opacity = 0.3
            
            if 'type' in attr:
                edge_label = attr.get('label', '')
                temp_graph.edges[u, v]['label'] = ''
                temp_graph.edges[u, v]['width'] = 2
                
                if attr['type'] == 'relation':
                    # Check if this relation's label should be highlighted
                    if edge_label in highlight_relations:
                        opacity = 1.0
                    
                    base_color = '#FF5722'
                    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
                    temp_graph.edges[u, v]['color'] = f'rgba({r}, {g}, {b}, {opacity})'
                    
                elif attr['type'] == 'connection':
                    # Check if this connection's label should be highlighted
                    if edge_label in highlight_connections:
                        opacity = 1.0
                    
                    base_color = '#9E9E9E'
                    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
                    temp_graph.edges[u, v]['color'] = f'rgba({r}, {g}, {b}, {opacity})'
        
        net.from_nx(temp_graph)
        
        net.set_options("""
        const options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            },
            "nodes": {
                "font": {
                    "size": 12,
                    "face": "Tahoma"
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": false
                    }
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            }
        }
        """)
        
        net.width = "100%"
        net.height = "1200px"

        net.save_graph(save_pth)
        self.logger.info(f"Save Graph HTML File to {save_pth}")