from .base import ChunkerBase

from openai import OpenAI
from typing import List
from nltk.tokenize import sent_tokenize

class OpenaiAgentChunker(ChunkerBase):
    def __init__(self, base_url, api_key, llm_model_name, context_sentence: int, time_out=120, re_chunk_times = 2):
        self.base_url = base_url
        self.llm_model_name = llm_model_name
        self.context_sentence = context_sentence
        self.api_key = api_key
        self.re_chunk_times = re_chunk_times

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=time_out
        )

        self.prompt = """
I'm going to give you two sentences now. 
You need to decide if they have a strong connection, meaning you should judge whether the second sentence belongs in the same paragraph as the first one. 
If it does, reply with 'y'; if not, reply with 'n'. When you respond, just use one letter, either 'y' or 'n'.  
"""
        

    def generate_response(self, system_prompt, chat_history):
        messages = []
        if system_prompt:
            messages = [{"role":"system","content":system_prompt}] + chat_history
        else:
            messages = chat_history

        
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages
        )
        return response.choices[0].message.content
    

    def get_lm_response(self, sentence1:str, sentence2:str, max_tries = 3):
        msg = self.prompt + "\nFirst sentence: \n" +sentence1 + "\n" + "\nSecond sentence:" + sentence2
        tries = 0
        while tries < max_tries:
            response = self.generate_response("",[{"role":"user","content":msg}])
            if response == "y":
                return True
            elif response == "n":
                return False
        raise Exception("lm chunking: expect an answer 'y' or 'n'.")

    
    def split_text_by_sentences(self, text: str, language) -> List[str]:
        """
        Split the text into sentences using nltk.tokenize.sent_tokenize.
        :param text: Input text
        :return: List of segmented sentences
        """
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


    def lm_chunk_text(self, sentences: List[str], context_len):
        chunks = []
        sentence_now = ""
        for iter in range(len(sentences)):
            if not sentence_now:
                sentence_now = sentences[iter]
            else:
                if self.get_lm_response(sentence_now, sentences[iter]):
                    sentence_now = sentence_now + sentences[iter]
                else:
                    for i in range(context_len):
                        sentence_now = sentence_now +sentences[iter+i+1]
                    chunks.append(sentence_now)
                    sentence_now = sentences[iter]

        return chunks
    

    def chunk_text(self, text:str, language = "en") -> List[str]:
        chunks = self.split_text_by_sentences(text, language)
        for i in range(self.re_chunk_times-1):
            chunks = self.lm_chunk_text(chunks, 0)
        chunks = self.lm_chunk_text(chunks, self.context_sentence)

        return chunks