from .base import ChunkerBase
from typing import List
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

class NaiveChunker(ChunkerBase):
    def __init__(self, model_name, model_cache_dir, context_sentence: int=1, max_token_length = 1200):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = model_cache_dir)
        self.max_token_length = max_token_length

        self.context_sentence = context_sentence

    
    def split_text_by_sentences(self, text: str, language) -> List[str]:
        if language == 'en':
            segments = sent_tokenize(text)
        elif language == 'zh':
            segments = []
            temp_sentence = ""
            for char in text:
                temp_sentence += char
                if char in ["。", "！", "？", "；"]:
                    segments.append(temp_sentence.strip())
                    temp_sentence = ""
            if temp_sentence:
                segments.append(temp_sentence.strip())
        else:
            raise Exception("Error in ppl chunking! No such language.")
        return [item for item in segments if item.strip()]
    
    def chunk_text(self, text:str, language = "en") -> List[str]:
        sentences = self.split_text_by_sentences(text, language)

        chunk_splite_num = []
        current_token_count = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_token_count = len(sentence_tokens)
            
            if current_token_count + sentence_token_count > self.max_token_length:
                chunk_splite_num.append(i)
                current_token_count = 0

            current_token_count += sentence_token_count
            
        
        if not chunk_splite_num:
            return [" ".join(sentences)]
        
        if chunk_splite_num[-1] < len(sentences):
            chunk_splite_num.append(len(sentences))

        chunks = []
        last_split_num = 0
        for split_iter in chunk_splite_num:
            chunks.append(" ".join(sentences[last_split_num:(split_iter+self.context_sentence)]))
            last_split_num = split_iter
            
        return chunks