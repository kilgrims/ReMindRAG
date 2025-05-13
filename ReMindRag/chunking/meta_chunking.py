from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from .base import ChunkerBase

class MetaChunker(ChunkerBase):
    def __init__(self, model_name_or_path: str, model_cache_dir:str, device: str = 'auto', threshold = 0.5, re_chunk_times = 2, chunk_batch_size = 64, context_sentence: int = 1) -> None:
        """
        Initialize the Chunking class.
        :param model_name_or_path: Path or name of the pre-trained model
        :param device: Device type ('cpu' or 'cuda')
        :param threshold: Threshold to determine minima points
        """
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir = model_cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir = model_cache_dir, trust_remote_code=True, device_map=device)
        self.model.eval()
        self.threshold =threshold
        self.re_chunk_times = re_chunk_times
        self.chunk_batch_size = chunk_batch_size
        self.context_sentence = context_sentence

    def get_ppl_batch(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None
    ):
        """
        Calculate the perplexity of the input text.
        :param input_ids: Input token IDs
        :param attention_mask: Attention mask
        :param past_key_values: Cached key-value pairs
        :param return_kv: Whether to return cached key-value pairs
        :param end: End position for perplexity calculation
        :return: Perplexity value (and cached key-value pairs if return_kv is True)
        """
        past_length = 0
        if end is None:
            end = input_ids.shape[1]
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1: end].contiguous()
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        res = loss
        return (res, past_key_values) if return_kv else res

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
        return [item for item in segments if item.strip()]  # Filter out empty sentences

    def calculate_ppl_for_sentences(self, segments: List[str], batch_size = 100) -> List[float]:
        first_cluster_ppl = []
        
        for i in range(0, len(segments), batch_size):
            batch_segments = segments[i:i+batch_size]
            
            len_sentences = []
            input_ids = torch.tensor([[]], device=self.model.device, dtype=torch.long)
            attention_mask = torch.tensor([[]], device=self.model.device, dtype=torch.long)
            
            for context in batch_segments:
                tokenized_text = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                input_id = tokenized_text["input_ids"].to(self.model.device)
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                len_sentences.append(input_id.shape[1])
                attention_mask_tmp = tokenized_text["attention_mask"].to(self.model.device)
                attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
            
            with torch.no_grad():
                loss = self.get_ppl_batch(input_ids, attention_mask)
            
            index = 0
            for j in range(len(len_sentences)):
                if i + j == 0:
                    seg_loss = loss[0:len_sentences[j]-1].mean().item()
                    index += len_sentences[j] - 1
                else:
                    seg_loss = loss[index:index+len_sentences[j]].mean().item()
                    index += len_sentences[j]
                first_cluster_ppl.append(seg_loss)
        
        return first_cluster_ppl

    def find_minima_indices(self, values: List[float]) -> List[int]:
        """
        Find the indices of local minima in the perplexity values.
        :param values: List of perplexity values
        :return: List of indices of local minima
        """
        minima_indices = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                if (values[i - 1] - values[i] >= self.threshold) or (values[i + 1] - values[i] >= self.threshold):
                    minima_indices.append(i)
        return minima_indices

    def meta_chunk(self, sentences: List[str], context_sentence = 1, chunk_batch_size = 100) -> List[str]:
        # Calculate perplexity for each sentence
        first_cluster_ppl = self.calculate_ppl_for_sentences(sentences, chunk_batch_size)

        # Find local minima points
        minima_indices = self.find_minima_indices(first_cluster_ppl)

        # Chunk the text based on local minima points
        final_chunks = []
        split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1] + context_sentence
            chunk = ''.join(sentences[start:end])
            final_chunks.append(chunk)

        return final_chunks
    

    def chunk_text(self, text:str, language = "en") -> List[str]:
        chunk_batch_size = self.chunk_batch_size

        chunks = self.split_text_by_sentences(text, language)

        for i in range(self.re_chunk_times-1):
            chunks = self.meta_chunk(chunks, 0, chunk_batch_size)
            chunk_batch_size = int(chunk_batch_size/3)
        chunks = self.meta_chunk(chunks, self.context_sentence, chunk_batch_size)

        return chunks