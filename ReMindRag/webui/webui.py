from flask import Flask, render_template, request, jsonify, url_for
import os
import numpy as np
from ..rag_main import ReMindRag


def launch_webui(rag_instance:ReMindRag, host='127.0.0.1', port=5000, debug=True):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    app = Flask(__name__)
    

    app.config.update(
        TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates"),
        STATIC_FOLDER = os.path.join(BASE_DIR, "temp"),
        CHROMA_PATH=os.environ.get(rag_instance.save_dir)
    )
    app.template_folder = app.config['TEMPLATE_FOLDER']
    app.static_folder = app.config["STATIC_FOLDER"]

    collections = {
        "entity": rag_instance.database.client.get_or_create_collection(
            name="entity",
            embedding_function=None
        ),
        "chunk": rag_instance.database.client.get_or_create_collection(
            name="chunk",
            embedding_function=None
        ),
        "relation": rag_instance.database.client.get_or_create_collection(
            name="relation",
            embedding_function=None
        ),
        "connection": rag_instance.database.client.get_or_create_collection(
            name="connection",
            embedding_function=None
        )
    }

    @app.route('/')
    def index():
        return render_template('index.html', collections=collections.keys())

    @app.route('/view/<collection_name>', methods=['GET', 'POST'])
    def view_collection(collection_name):
        print(f"view: {collection_name}")

        if collection_name not in collections:
            return "Collection not found", 404
        
        collection = collections[collection_name]
        query_text = ""
        n_results = 10
        
        try:
            if request.method == 'POST':
                query_text = request.form.get('query_text', '')
                n_results = int(request.form.get('n_results', 10))
                
                if query_text:
                    query_embedding = rag_instance.embedding.sentence_embedding(query_text)
                    
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=['metadatas', 'documents', 'embeddings']
                    )
                    
                    items = []
                    for i in range(len(results['ids'][0])):
                        embedding_vector = results.get('embeddings', [[]])[0][i]
                        if isinstance(embedding_vector, np.ndarray):
                            embedding_vector = embedding_vector.tolist()
                        
                        document_text = results['documents'][0][i] if 'documents' in results else ""
                        chunk_size = len(document_text)
                        
                        items.append((
                            results['ids'][0][i],
                            embedding_vector,
                            results['metadatas'][0][i] if 'metadatas' in results else {},
                            document_text,
                            chunk_size
                        ))
                else:
                    results = collection.get(include=['metadatas', 'documents', 'embeddings'])
                    
                    items = []
                    for i in range(len(results['ids'])):
                        embedding_vector = results.get('embeddings', [])[i]
                        if isinstance(embedding_vector, np.ndarray):
                            embedding_vector = embedding_vector.tolist()
                        
                        document_text = results['documents'][i] if 'documents' in results else ""
                        chunk_size = len(document_text)
                        
                        items.append((
                            results['ids'][i],
                            embedding_vector,
                            results['metadatas'][i] if 'metadatas' in results else {},
                            document_text,
                            chunk_size
                        ))
            else:
                results = collection.get(include=['metadatas', 'documents', 'embeddings'])
                
                items = []
                for i in range(len(results['ids'])):
                    embedding_vector = results.get('embeddings', [])[i]
                    if isinstance(embedding_vector, np.ndarray):
                        embedding_vector = embedding_vector.tolist()
                    
                    document_text = results['documents'][i] if 'documents' in results else ""
                    chunk_size = len(document_text)
                    
                    items.append((
                        results['ids'][i],
                        embedding_vector,
                        results['metadatas'][i] if 'metadatas' in results else {},
                        document_text,
                        chunk_size
                    ))
            
            return render_template(
                'collection.html',
                collection_name=collection_name,
                items=items,
                query_text=query_text,
                n_results=n_results
            )
        except Exception as e:
            return f"Error retrieving data: {str(e)}", 500

    @app.route('/search', methods=['GET'])
    def search_page():
        return render_template('search.html', collections=collections.keys())

    @app.route('/api/search/<collection_name>', methods=['POST'])
    def api_search(collection_name):
        if collection_name not in collections:
            return {"error": "Collection not found"}, 404
        
        collection = collections[collection_name]
        data = request.json
        
        query_text = data.get('query_text', '')
        n_results = data.get('n_results', 10)
        
        try:
            if query_text:
                query_embedding = rag_instance.embedding.sentence_embedding(query_text)
                
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=['metadatas', 'documents', 'embeddings']
                )

                response = {
                    "results": []
                }
                
                for i in range(len(results['ids'][0])):
                    embedding_vector = results.get('embeddings', [[]])[0][i]
                    if isinstance(embedding_vector, np.ndarray):
                        embedding_vector = embedding_vector.tolist()
                    
                    document_text = results['documents'][0][i] if 'documents' in results else ""
                    chunk_size = len(document_text)
                    
                    response["results"].append({
                        "id": results['ids'][0][i],
                        "embedding": embedding_vector,
                        "metadata": results['metadatas'][0][i] if 'metadatas' in results else {},
                        "document": document_text,
                        "chunk_size": chunk_size
                    })
                    
                return response
            else:
                return {"error": "No query text provided"}, 400
        except Exception as e:
            return {"error": str(e)}, 500


    @app.route('/process_query', methods=['POST'])
    def process_query():
        print("get query")
        data = request.get_json()
        query = data.get('query', '')
        
        
        html_name = ""
        rag_instance.refresh_kg()
        if not query:
            html_name = "graph_base.html"
            html_pth = os.path.join(BASE_DIR, "temp", html_name)
            rag_instance.export_kg_as_pyvis(html_pth)
        else:
            html_name = "graph_base_query.html"
            html_pth = os.path.join(BASE_DIR, "temp", html_name)
            rag_instance.kg.save_as_pyvis_for_quick_query(html_pth, query)


        return jsonify({
            'file_url': url_for('static', filename=f'{os.path.basename(html_name)}', _external=True)
        })


    app.run(host=host, port=port, debug=debug)