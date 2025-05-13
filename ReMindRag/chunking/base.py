from abc import ABC, abstractmethod
from typing import List

class ChunkerBase(ABC):
    @abstractmethod
    def chunk_text(self, text:str, language:str) -> List[str]:
        pass