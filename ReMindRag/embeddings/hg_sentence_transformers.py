from .base import EmbeddingBase
from sentence_transformers import SentenceTransformer


class HgEmbedding(EmbeddingBase):
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        self.model = SentenceTransformer(model_name_or_path=model_name, cache_folder=cache_dir, trust_remote_code=True)

    def sentence_embedding(self, sentence):
        embedding = self.model.encode([sentence])
        return embedding[0]

    def sentence_list_embedding(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
    
    def get_hidden_state_size(self) -> int:
        example_embedding = self.sentence_embedding("example")
        return len(example_embedding)