from .base import EmbeddingBase
from openai import OpenAI
import numpy as np

class OpenaiEmbedding(EmbeddingBase):
    def __init__(self, base_url, api_key, embedding_model_name):
        self.base_url = base_url
        self.embedding_model_name = embedding_model_name
        self.api_key = api_key

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def sentence_embedding(self, sentence):
        response = self.client.embeddings.create(
            input=sentence,
            model=self.embedding_model_name
        )

        embedding = np.array(response.data[0].embedding)

        return embedding
    
    def sentence_list_embedding(self, sentences):
        embeddings = []
        for sentence_iter in sentences:
            response = self.client.embeddings.create(
                input=sentence_iter,
                model=self.embedding_model_name
            )
            embeddings.append(response.data[0].embedding)

        embeddings = np.array(embeddings)
        return embeddings
    
    def get_hidden_state_size(self) -> int:
        example_embedding = self.sentence_embedding("example")
        return len(example_embedding)