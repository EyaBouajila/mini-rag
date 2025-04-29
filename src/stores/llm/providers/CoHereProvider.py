from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging
import time
from typing import Union, List

class CoHereProvider(LLMInterface):
    def __init__(self, api_key: str, base_url: str = None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.api_key = api_key
        self.base_url = base_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(
            api_key=self.api_key,
            base_url=self.base_url
        )


        self.enums = CoHereEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None, temperature: float = None):
        
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=CoHereEnums.USER.value)
        )

        response = self.client.chat(
            model = self.generation_model_id,
            chat_history = chat_history,
            message = self.process_text(prompt),
            temperature = temperature,
            max_tokens= max_output_tokens

        )

        if not response or not response.text:
            self.logger.error("Error while generating text with CoHere")
            return None
        
        return response.text
    
    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None
        
        if isinstance(text, str):
            text = [text]
        
        input_type = CoHereEnums.DOCUMENT
        if document_type == DocumentTypeEnum.QUERY:
            input_type = CoHereEnums.QUERY

######## test 1 : undefinite looping until results (not production-friendly) ########
        # delay_seconds = 10
        # attempt = 0

        # while True:
        #     attempt += 1
        #     try:
        #         response = self.client.embed(
        #             model=self.embedding_model_id,
        #             texts=[self.process_text(text)],
        #             input_type=input_type,
        #             embedding_types=['float']
        #         )

        #         if response and response.embeddings and response.embeddings.float:
        #             self.logger.info(f"Embedding successful on attempt {attempt}")
        #             return response.embeddings.float[0]

        #         self.logger.warning(f"Attempt {attempt}: Empty embedding response")
        #     except Exception as e:
        #         self.logger.error(f"Attempt {attempt}: Exception during embedding: {e}")

        #     self.logger.info(f"Retrying in {delay_seconds} seconds...")
        #     time.sleep(delay_seconds)

######## test 2 : definite looping until results (not production-friendly & unknown max_attempts) ########
        # max_attempts = 20
        # delay_seconds = 2

        # for attempt in range(1, max_attempts + 1):
        #     try:
        #         response = self.client.embed(
        #             model=self.embedding_model_id,
        #             texts=[self.process_text(text)],
        #             input_type=input_type,
        #             embedding_types=['float']
        #         )

        #         if response and response.embeddings and response.embeddings.float:
        #             return response.embeddings.float[0]

        #         self.logger.warning(f"Empty embedding response on attempt {attempt}")
        #     except Exception as e:
        #         self.logger.error(f"Exception during embedding (attempt {attempt}): {e}")

        #     # Wait before retrying
        #     time.sleep(delay_seconds)

        # self.logger.error("Failed to get embedding after multiple attempts")
        # return None

        try:
            # may have sent a request when the CoHere server returned an empty response (e.g. due to rate limiting, timeout, or internal server hiccup).
            # maybe text: Union[str, List[str]] would solve it (instead of text: str)
            # also check nested sessions 
            response = self.client.embed(
                model = self.embedding_model_id,
                texts = [self.process_text(t) for t in text],
                input_type = input_type,
                embedding_types = ['float']
            )

        except Exception as e:
            self.logger.error(f"Exception during embedding: {e}")
            return None


        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Error while embedding text with CoHere")
            return None
        
        return response.embeddings[0]
        # # return response.embeddings or response.embeddings.float[0]


    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "message": self.process_text(prompt)
        }
    
    