from abc import ABC, abstractmethod
import numpy as np

class EmbeddingBase(ABC):
    @abstractmethod
    def sentence_embedding(self, sentence:str) -> np.array:
        pass

    @abstractmethod
    def sentence_list_embedding(self, sentences:list[str]) -> np.array:
        pass

    @abstractmethod
    def get_hidden_state_size(self) -> int:
        pass