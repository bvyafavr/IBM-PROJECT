# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# import torch
# import numpy as np
# import re
# import time
# import asyncio

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app = FastAPI(title="T5 Paraphrasing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# # Request/Response Schemas
# class ParaphraseRequest(BaseModel):
#     text: str
#     strength: Literal["light", "medium", "strong"] = "medium"

# class ParaphraseResponse(BaseModel):
#     paraphrased_text: str
#     model_used: str
#     similarity: float
#     processing_time: float

# # Load T5 Paraphraser Model
# @lru_cache(maxsize=1)
# def load_t5():
#     tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
#     model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser").to(device)
#     return tokenizer, model

# @lru_cache(maxsize=1)
# def load_sentence_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # Utils
# def clean_text(text: str) -> str:
#     return re.sub(r'\s+', ' ', text).strip()

# def calculate_similarity(original: str, paraphrased: str) -> float:
#     model = load_sentence_model()
#     emb = model.encode([original, paraphrased])
#     return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))

# # Paraphrasing logic
# async def t5_paraphrase(text: str, strength: str = "medium") -> str:
#     tokenizer, model = load_t5()

#     prompt = f"paraphrase the sentence without turning it into a question  : {text} </s>"

#     strength_params = {
#         "light": {"num_beams": 4, "temperature": 0.7},
#         "medium": {"num_beams": 5, "temperature": 1.0},
#         "strong": {"num_beams": 6, "temperature": 1.3},
#     }

#     params = strength_params.get(strength, strength_params["medium"])

#     input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)

#     loop = asyncio.get_event_loop()
#     output = await loop.run_in_executor(
#         None,
#         lambda: model.generate(
#             input_ids=input_ids,
#             max_length=256,
#             num_beams=params["num_beams"],
#             temperature=params["temperature"],
#             top_k=50,
#             top_p=0.95,
#             early_stopping=True,
#             do_sample=True,
#             num_return_sequences=1
#         )
#     )

#     paraphrased = tokenizer.decode(output[0], skip_special_tokens=True)
#     return clean_text(paraphrased)

# # API Route
# @app.post("/api/paraphrase", response_model=ParaphraseResponse)
# async def paraphrase(req: ParaphraseRequest):
#     start = time.time()
#     max_attempts = 5
#     final_result = req.text
#     similarity = 1.0

#     for _ in range(max_attempts):
#         result = await t5_paraphrase(req.text, req.strength)
#         similarity = calculate_similarity(req.text, result)

#         if similarity <= 0.83:
#             final_result = result
#             break
#         else:
#             final_result = result  # fallback even if it's too similar

#     return ParaphraseResponse(
#         paraphrased_text=final_result,
#         model_used="ramsrigouthamg/t5_paraphraser",
#         similarity=round(similarity, 4),
#         processing_time=round(time.time() - start, 3)
#     )

# @app.get("/")
# def root():
#     return {"message": "T5 Paraphrasing API is running!"}




# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# import torch
# import numpy as np
# import re
# import time
# import asyncio

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app = FastAPI(title="T5 Paraphrasing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# # Request/Response Schemas
# class ParaphraseRequest(BaseModel):
#     text: str
#     strength: Literal["light", "medium", "strong"] = "medium"

# class ParaphraseResponse(BaseModel):
#     paraphrased_text: str
#     model_used: str
#     similarity: float
#     processing_time: float

# # Load T5 Paraphraser Model
# @lru_cache(maxsize=1)
# def load_t5():
#     tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
#     model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)
#     return tokenizer, model

# @lru_cache(maxsize=1)
# def load_sentence_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # Utils
# def clean_text(text: str) -> str:
#     return re.sub(r'\s+', ' ', text).strip()

# def calculate_similarity(original: str, paraphrased: str) -> float:
#     model = load_sentence_model()
#     emb = model.encode([original, paraphrased])
#     return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))

# # Paraphrasing logic
# async def t5_paraphrase(text: str, strength: str = "medium") -> str:
#     tokenizer, model = load_t5()
    
#     prompt = f"paraphrase: {text} </s>"
    
#     strength_params = {
#         "light": {"num_beams": 4, "temperature": 0.7},
#         "medium": {"num_beams": 5, "temperature": 1.0},
#         "strong": {"num_beams": 6, "temperature": 1.3},
#     }
    
#     params = strength_params.get(strength, strength_params["medium"])
    
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True).to(device)
    
#     loop = asyncio.get_event_loop()
#     output = await loop.run_in_executor(
#         None,
#         lambda: model.generate(
#             input_ids=input_ids,
#             max_length=256,
#             num_beams=params["num_beams"],
#             temperature=params["temperature"],
#             top_k=50,
#             top_p=0.95,
#             early_stopping=True,
#             do_sample=True,
#             num_return_sequences=1
#         )
#     )
    
#     paraphrased = tokenizer.decode(output[0], skip_special_tokens=True)
#     return clean_text(paraphrased)

# # API Route
# @app.post("/api/paraphrase", response_model=ParaphraseResponse)
# async def paraphrase(req: ParaphraseRequest):
#     start = time.time()
#     max_attempts = 5
#     final_result = req.text
#     similarity = 1.0
    
#     for _ in range(max_attempts):
#         result = await t5_paraphrase(req.text, req.strength)
#         similarity = calculate_similarity(req.text, result)
        
#         if similarity <= 0.83:
#             final_result = result
#             break
#         else:
#             final_result = result  # fallback even if it's too similar
    
#     return ParaphraseResponse(
#         paraphrased_text=final_result,
#         model_used="Vamsi/T5_Paraphrase_Paws",
#         similarity=round(similarity, 4),
#         processing_time=round(time.time() - start, 3)
#     )

# @app.get("/")
# def root():
#     return {"message": "T5 Paraphrasing API is running!"}




# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# import torch
# import numpy as np
# import re
# import time
# import asyncio

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app = FastAPI(title="T5 Paraphrasing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# # Request/Response Schemas
# class ParaphraseRequest(BaseModel):
#     text: str
#     strength: Literal["light", "medium", "strong"] = "medium"

# class ParaphraseResponse(BaseModel):
#     paraphrased_text: str
#     model_used: str
#     similarity: float
#     processing_time: float

# # Load T5 Paraphraser Model
# @lru_cache(maxsize=1)
# def load_t5():
#     tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
#     model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase").to(device)
#     return tokenizer, model

# @lru_cache(maxsize=1)
# def load_sentence_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # Utils
# def clean_text(text: str) -> str:
#     return re.sub(r'\s+', ' ', text).strip()

# def is_question(text: str) -> bool:
#     text = text.strip()
#     question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom']
#     return text.endswith('?') or any(text.lower().startswith(word) for word in question_words)

# def calculate_similarity(original: str, paraphrased: str) -> float:
#     model = load_sentence_model()
#     emb = model.encode([original, paraphrased])
#     return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))

# # Paraphrasing logic
# async def t5_paraphrase(text: str, strength: str = "medium") -> str:
#     tokenizer, model = load_t5()
    
#     # Pegasus doesn't need special prompt format
#     input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
#     strength_params = {
#         "light": {"num_beams": 4, "temperature": 0.8, "top_p": 0.9},
#         "medium": {"num_beams": 5, "temperature": 1.2, "top_p": 0.85},
#         "strong": {"num_beams": 6, "temperature": 1.5, "top_p": 0.8},
#     }
    
#     params = strength_params.get(strength, strength_params["medium"])
    
#     loop = asyncio.get_event_loop()
#     output = await loop.run_in_executor(
#         None,
#         lambda: model.generate(
#             input_ids=input_ids,
#             max_length=512,
#             num_beams=params["num_beams"],
#             temperature=params["temperature"],
#             top_k=50,
#             top_p=params["top_p"],
#             early_stopping=True,
#             do_sample=True,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2
#         )
#     )
    
#     paraphrased = tokenizer.decode(output[0], skip_special_tokens=True)
#     return clean_text(paraphrased)

# # API Route
# @app.post("/api/paraphrase", response_model=ParaphraseResponse)
# async def paraphrase(req: ParaphraseRequest):
#     start = time.time()
#     max_attempts = 5
#     final_result = req.text
#     similarity = 1.0
    
#     for attempt in range(max_attempts):
#         result = await t5_paraphrase(req.text, req.strength)
#         result_clean = clean_text(result)
        
#         # Skip if it's a question
#         if is_question(result_clean):
#             continue
            
#         similarity = calculate_similarity(req.text, result_clean)
        
#         if similarity <= 0.83:
#             final_result = result_clean
#             break
#         else:
#             final_result = result_clean  # fallback even if it's too similar
    
#     return ParaphraseResponse(
#         paraphrased_text=final_result,
#         model_used="tuner007/pegasus_paraphrase",
#         similarity=round(similarity, 4),
#         processing_time=round(time.time() - start, 3)
#     )

# @app.get("/")
# def root():
#     return {"message": "T5 Paraphrasing API is running!"}





# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# import torch
# import numpy as np
# import re
# import time
# import asyncio

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app = FastAPI(title="T5 Paraphrasing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# # Request/Response Schemas
# class ParaphraseRequest(BaseModel):
#     text: str
#     strength: Literal["light", "medium", "strong"] = "medium"
#     mode: str = "standard"  # Accept mode but ignore it

# class ParaphraseResponse(BaseModel):
#     paraphrased_text: str
#     model_used: str
#     similarity: float
#     processing_time: float

# # Load T5 Paraphraser Model
# @lru_cache(maxsize=1)
# def load_t5():
#     tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
#     return tokenizer, model

# @lru_cache(maxsize=1)
# def load_sentence_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # Utils
# def clean_text(text: str) -> str:
#     return re.sub(r'\s+', ' ', text).strip()

# def is_question(text: str) -> bool:
#     text = text.strip()
#     question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom']
#     return text.endswith('?') or any(text.lower().startswith(word) for word in question_words)

# def split_into_sentences(text: str) -> list:
#     # Simple sentence splitting
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     return [s.strip() for s in sentences if s.strip()]

# def combine_sentences(sentences: list) -> str:
#     return ' '.join(sentences)

# def calculate_similarity(original: str, paraphrased: str) -> float:
#     model = load_sentence_model()
#     emb = model.encode([original, paraphrased])
#     return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))

# # Paraphrasing logic
# async def t5_paraphrase_single(text: str, strength: str = "medium") -> str:
#     tokenizer, model = load_t5()
    
#     # This model uses a specific prompt format
#     prompt = f"paraphrase: {text}"
    
#     strength_params = {
#         "light": {"num_beams": 3, "temperature": 0.7, "repetition_penalty": 1.1},
#         "medium": {"num_beams": 4, "temperature": 1.0, "repetition_penalty": 1.2},
#         "strong": {"num_beams": 5, "temperature": 1.3, "repetition_penalty": 1.3},
#     }
    
#     params = strength_params.get(strength, strength_params["medium"])
    
#     try:
#         input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
        
#         loop = asyncio.get_event_loop()
#         output = await loop.run_in_executor(
#             None,
#             lambda: model.generate(
#                 input_ids=input_ids,
#                 max_length=400,
#                 num_beams=params["num_beams"],
#                 temperature=params["temperature"],
#                 repetition_penalty=params["repetition_penalty"],
#                 top_k=50,
#                 top_p=0.9,
#                 early_stopping=True,
#                 do_sample=True,
#                 num_return_sequences=1,
#                 no_repeat_ngram_size=2
#             )
#         )
        
#         paraphrased = tokenizer.decode(output[0], skip_special_tokens=True)
#         return clean_text(paraphrased)
#     except Exception as e:
#         print(f"Error in paraphrasing: {e}")
#         return text

# async def t5_paraphrase(text: str, strength: str = "medium") -> str:
#     # Split long text into sentences and paraphrase each
#     sentences = split_into_sentences(text)
    
#     if len(sentences) <= 1:
#         return await t5_paraphrase_single(text, strength)
    
#     paraphrased_sentences = []
#     for sentence in sentences:
#         if len(sentence.strip()) > 5:  # Only paraphrase meaningful sentences
#             paraphrased = await t5_paraphrase_single(sentence, strength)
#             paraphrased_sentences.append(paraphrased)
#         else:
#             paraphrased_sentences.append(sentence)
    
#     return combine_sentences(paraphrased_sentences)

# # API Route
# @app.post("/api/paraphrase", response_model=ParaphraseResponse)
# async def paraphrase(req: ParaphraseRequest):
#     start = time.time()
#     max_attempts = 5
#     final_result = req.text
#     similarity = 1.0
    
#     for attempt in range(max_attempts):
#         result = await t5_paraphrase(req.text, req.strength)
#         result_clean = clean_text(result)
        
#         # Skip if it's a question
#         if is_question(result_clean):
#             continue
            
#         similarity = calculate_similarity(req.text, result_clean)
        
#         if similarity <= 0.83:
#             final_result = result_clean
#             break
#         else:
#             final_result = result_clean  # fallback even if it's too similar
    
#     return ParaphraseResponse(
#         paraphrased_text=final_result,
#         model_used="humarin/chatgpt_paraphraser_on_T5_base",
#         similarity=round(similarity, 4),
#         processing_time=round(time.time() - start, 3)
#     )

# @app.get("/")
# def root():
#     return {"message": "T5 Paraphrasing API is running!"}



# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# import torch
# import numpy as np
# import re
# import time
# import asyncio

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app = FastAPI(title="T5 Paraphrasing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# # Request/Response Schemas
# class ParaphraseRequest(BaseModel):
#     text: str
#     strength: Literal["light", "medium", "strong"] = "medium"
#     mode: str = "standard"  # Accept mode but ignore it

# class ParaphraseResponse(BaseModel):
#     paraphrased_text: str
#     model_used: str
#     similarity: float
#     processing_time: float

# # Load T5 Paraphraser Model
# @lru_cache(maxsize=1)
# def load_t5():
#     tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
#     return tokenizer, model

# @lru_cache(maxsize=1)
# def load_sentence_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # Utils
# def clean_text(text: str) -> str:
#     # Minimal cleaning to preserve natural imperfections
#     text = re.sub(r'\s{3,}', ' ', text)  # Only fix excessive spaces
#     return text.strip()

# def add_human_touches(text: str) -> str:
#     # Add subtle human-like imperfections
#     import random
    
#     # Randomly remove space after comma (like your example)
#     if random.random() < 0.3:
#         text = re.sub(r', ', ',', text, count=1)
    
#     # Occasionally use contractions
#     contractions = {
#         ' will ': ' will ', ' would ': ' would ', ' cannot ': ' cannot ',
#         ' do not ': ' don\'t ', ' does not ': ' doesn\'t ', ' did not ': ' didn\'t ',
#         ' are not ': ' aren\'t ', ' is not ': ' isn\'t ', ' was not ': ' wasn\'t '
#     }
    
#     for full, short in contractions.items():
#         if random.random() < 0.2 and full in text:
#             text = text.replace(full, short, 1)
    
#     return text

# def is_question(text: str) -> bool:
#     text = text.strip()
#     question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom']
#     return text.endswith('?') or any(text.lower().startswith(word) for word in question_words)

# def split_into_sentences(text: str) -> list:
#     # Simple sentence splitting
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     return [s.strip() for s in sentences if s.strip()]

# def combine_sentences(sentences: list) -> str:
#     return ' '.join(sentences)

# def calculate_similarity(original: str, paraphrased: str) -> float:
#     model = load_sentence_model()
#     emb = model.encode([original, paraphrased])
#     return float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))

# # Paraphrasing logic
# async def t5_paraphrase_single(text: str, strength: str = "medium") -> str:
#     tokenizer, model = load_t5()
    
#     # This model uses a specific prompt format
#     prompt = f"paraphrase: {text}"
    
#     # More human-like parameters - higher temperature, more randomness
#     strength_params = {
#         "light": {"num_beams": 2, "temperature": 1.2, "repetition_penalty": 1.05, "top_p": 0.95},
#         "medium": {"num_beams": 3, "temperature": 1.5, "repetition_penalty": 1.1, "top_p": 0.9},
#         "strong": {"num_beams": 4, "temperature": 1.8, "repetition_penalty": 1.15, "top_p": 0.85},
#     }
    
#     params = strength_params.get(strength, strength_params["medium"])
    
#     try:
#         input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
        
#         loop = asyncio.get_event_loop()
#         output = await loop.run_in_executor(
#             None,
#             lambda: model.generate(
#                 input_ids=input_ids,
#                 max_length=400,
#                 num_beams=params["num_beams"],
#                 temperature=params["temperature"],
#                 repetition_penalty=params["repetition_penalty"],
#                 top_k=40,  # Lower for more variety
#                 top_p=params["top_p"],
#                 early_stopping=True,
#                 do_sample=True,
#                 num_return_sequences=1,
#                 no_repeat_ngram_size=2,
#                 pad_token_id=tokenizer.eos_token_id
#             )
#         )
        
#         paraphrased = tokenizer.decode(output[0], skip_special_tokens=True)
#         # Apply human-like touches
#         paraphrased = add_human_touches(clean_text(paraphrased))
#         return paraphrased
#     except Exception as e:
#         print(f"Error in paraphrasing: {e}")
#         return text

# async def t5_paraphrase(text: str, strength: str = "medium") -> str:
#     # Split long text into sentences and paraphrase each
#     sentences = split_into_sentences(text)
    
#     if len(sentences) <= 1:
#         return await t5_paraphrase_single(text, strength)
    
#     paraphrased_sentences = []
#     for sentence in sentences:
#         if len(sentence.strip()) > 5:  # Only paraphrase meaningful sentences
#             paraphrased = await t5_paraphrase_single(sentence, strength)
#             paraphrased_sentences.append(paraphrased)
#         else:
#             paraphrased_sentences.append(sentence)
    
#     return combine_sentences(paraphrased_sentences)

# # API Route
# @app.post("/api/paraphrase", response_model=ParaphraseResponse)
# async def paraphrase(req: ParaphraseRequest):
#     start = time.time()
#     max_attempts = 5
#     final_result = req.text
#     similarity = 1.0
    
#     for attempt in range(max_attempts):
#         result = await t5_paraphrase(req.text, req.strength)
#         result_clean = clean_text(result)
        
#         # Skip if it's a question
#         if is_question(result_clean):
#             continue
            
#         similarity = calculate_similarity(req.text, result_clean)
        
#         if similarity <= 0.83:
#             final_result = result_clean
#             break
#         else:
#             final_result = result_clean  # fallback even if it's too similar
    
#     return ParaphraseResponse(
#         paraphrased_text=final_result,
#         model_used="humarin/chatgpt_paraphraser_on_T5_base",
#         similarity=round(similarity, 4),
#         processing_time=round(time.time() - start, 3)
#     )

# @app.get("/")
# def root():
#     return {"message": "T5 Paraphrasing API is running!"}

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Literal
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# from functools import lru_cache
# import torch
# import numpy as np
# import re
# import time
# import asyncio
# import random

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app = FastAPI(title="T5 Paraphrasing API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
# )

# # Request/Response Schemas
# class ParaphraseRequest(BaseModel):
#     text: str
#     strength: Literal["light", "medium", "strong"] = "medium"
#     mode: str = "standard"  # Accept mode but ignore it

# class ParaphraseResponse(BaseModel):
#     paraphrased_text: str
#     model_used: str
#     similarity: float
#     processing_time: float

# # Load T5 Paraphraser Model
# @lru_cache(maxsize=1)
# def load_t5():
#     tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
#     return tokenizer, model

# @lru_cache(maxsize=1)
# def load_sentence_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# # Enhanced synonym replacement for more variation
# def advanced_synonym_replacement(text: str) -> str:
#     """More aggressive synonym replacement to reduce similarity"""
#     replacements = {
#         # Common words with multiple alternatives
#         'good': ['excellent', 'great', 'fine', 'nice', 'decent', 'solid'],
#         'bad': ['poor', 'terrible', 'awful', 'horrible', 'lousy'],
#         'big': ['large', 'huge', 'massive', 'enormous', 'substantial'],
#         'small': ['tiny', 'little', 'minor', 'compact', 'minimal'],
#         'important': ['crucial', 'vital', 'essential', 'significant', 'key'],
#         'help': ['assist', 'aid', 'support', 'facilitate'],
#         'make': ['create', 'produce', 'generate', 'build', 'construct'],
#         'get': ['obtain', 'acquire', 'receive', 'gain', 'secure'],
#         'use': ['utilize', 'employ', 'apply', 'implement'],
#         'show': ['demonstrate', 'display', 'reveal', 'exhibit'],
#         'find': ['discover', 'locate', 'identify', 'uncover'],
#         'think': ['believe', 'consider', 'feel', 'suppose'],
#         'know': ['understand', 'realize', 'recognize', 'comprehend'],
#         'want': ['desire', 'wish', 'need', 'require'],
#         'work': ['function', 'operate', 'perform', 'labor'],
#         'look': ['appear', 'seem', 'glance', 'observe'],
#         'try': ['attempt', 'endeavor', 'strive', 'effort'],
#         'give': ['provide', 'offer', 'supply', 'deliver'],
#         'take': ['grab', 'seize', 'accept', 'receive'],
#         'come': ['arrive', 'approach', 'emerge', 'appear'],
#         'go': ['proceed', 'travel', 'move', 'head'],
#         'different': ['various', 'distinct', 'separate', 'diverse'],
#         'many': ['numerous', 'several', 'multiple', 'various'],
#         'most': ['majority', 'bulk', 'primary'],
#         'also': ['additionally', 'furthermore', 'moreover', 'plus'],
#         'however': ['nevertheless', 'nonetheless', 'but', 'yet'],
#         'because': ['since', 'due to', 'as', 'owing to'],
#         'very': ['extremely', 'quite', 'really', 'highly'],
#         'really': ['truly', 'genuinely', 'actually', 'indeed'],
#         'often': ['frequently', 'regularly', 'commonly'],
#         'always': ['constantly', 'continuously', 'perpetually'],
#         'never': ['not ever', 'at no time', 'under no circumstances'],
#         'sometimes': ['occasionally', 'at times', 'now and then'],
#         'quickly': ['rapidly', 'swiftly', 'speedily', 'fast'],
#         'slowly': ['gradually', 'leisurely', 'steadily'],
#         'easily': ['effortlessly', 'simply', 'readily'],
#         'difficult': ['challenging', 'hard', 'tough', 'complex']
#     }
    
#     words = text.split()
#     result_words = []
    
#     for word in words:
#         clean_word = re.sub(r'[^\w]', '', word.lower())
#         if clean_word in replacements and random.random() < 0.4:  # 40% chance to replace
#             synonym = random.choice(replacements[clean_word])
#             # Preserve original case and punctuation
#             if word[0].isupper():
#                 synonym = synonym.capitalize()
#             # Add back punctuation
#             punct = re.sub(r'\w', '', word)
#             result_words.append(synonym + punct)
#         else:
#             result_words.append(word)
    
#     return ' '.join(result_words)

# def structural_variation(text: str) -> str:
#     """Apply structural changes to reduce similarity"""
#     # Convert passive to active voice patterns
#     passive_patterns = [
#         (r'(\w+) is (\w+ed) by (\w+)', r'\3 \2s \1'),
#         (r'(\w+) was (\w+ed) by (\w+)', r'\3 \2 \1'),
#         (r'(\w+) are (\w+ed) by (\w+)', r'\3 \2 \1'),
#     ]
    
#     for pattern, replacement in passive_patterns:
#         if random.random() < 0.3:
#             text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
#     # Sentence restructuring
#     if random.random() < 0.3:
#         # Move prepositional phrases
#         prep_pattern = r'^([^,]+), (in|on|at|by|with|for|during|after|before) ([^,]+), (.+)$'
#         match = re.match(prep_pattern, text)
#         if match:
#             main, prep, phrase, rest = match.groups()
#             text = f"{prep.capitalize()} {phrase}, {main.lower()}, {rest}"
    
#     return text

# # Utils
# def clean_text(text: str) -> str:
#     # More aggressive cleaning while preserving natural style
#     text = re.sub(r'\s{2,}', ' ', text)
#     text = re.sub(r'([.!?])\s*([.!?])', r'\1', text)  # Remove duplicate punctuation
#     return text.strip()

# def add_human_touches(text: str, strength: str = "medium") -> str:
#     """Enhanced human-like modifications with strength control"""
#     import random
    
#     # Adjust randomness based on strength
#     replacement_chance = {"light": 0.2, "medium": 0.35, "strong": 0.5}[strength]
    
#     # More extensive casual replacements
#     casual_replacements = {
#         'extremely': ['really', 'super', 'very'],
#         'significantly': ['quite a bit', 'a lot', 'considerably'],
#         'numerous': ['many', 'lots of', 'plenty of'],
#         'various': ['different', 'several', 'all sorts of'],
#         'demonstrate': ['show', 'prove', 'illustrate'],
#         'utilize': ['use', 'employ', 'make use of'],
#         'approximately': ['about', 'around', 'roughly'],
#         'furthermore': ['also', 'plus', 'what\'s more'],
#         'moreover': ['also', 'besides', 'on top of that'],
#         'however': ['but', 'though', 'yet'],
#         'therefore': ['so', 'thus', 'that\'s why'],
#         'nevertheless': ['still', 'even so', 'but'],
#         'subsequently': ['then', 'later', 'after that'],
#         'initially': ['at first', 'to start with', 'in the beginning'],
#         'ultimately': ['in the end', 'finally', 'eventually'],
#         'substantial': ['big', 'major', 'significant'],
#         'optimal': ['best', 'ideal', 'perfect'],
#         'facilitate': ['help', 'make easier', 'assist'],
#         'commence': ['start', 'begin', 'kick off'],
#         'terminate': ['end', 'stop', 'finish'],
#     }
    
#     # Apply replacements more aggressively
#     for formal, casuals in casual_replacements.items():
#         if random.random() < replacement_chance:
#             pattern = r'\b' + re.escape(formal) + r'\b'
#             if re.search(pattern, text, re.IGNORECASE):
#                 casual = random.choice(casuals)
#                 text = re.sub(pattern, casual, text, flags=re.IGNORECASE, count=1)
    
#     # Apply synonym replacement
#     text = advanced_synonym_replacement(text)
    
#     # Apply structural variations
#     text = structural_variation(text)
    
#     # More frequent punctuation changes
#     if random.random() < 0.4:
#         text = re.sub(r', ', ', ', text, count=random.randint(1, 2))
    
#     # More casual starters
#     if random.random() < 0.25:
#         casual_starters = ['Well, ', 'So, ', 'Actually, ', 'You know, ', 'Basically, ']
#         if not any(text.startswith(starter.strip()) for starter in casual_starters):
#             starter = random.choice(casual_starters)
#             text = starter + text.lower()
#             text = text[0].upper() + text[1:]
    
#     # More contractions
#     contractions = {
#         ' do not ': ' don\'t ', ' does not ': ' doesn\'t ', 
#         ' are not ': ' aren\'t ', ' is not ': ' isn\'t ',
#         ' have not ': ' haven\'t ', ' has not ': ' hasn\'t ',
#         ' will not ': ' won\'t ', ' would not ': ' wouldn\'t ',
#         ' cannot ': ' can\'t ', ' could not ': ' couldn\'t ',
#         ' should not ': ' shouldn\'t ', ' must not ': ' mustn\'t '
#     }
    
#     for full, short in contractions.items():
#         if random.random() < 0.35 and full in text.lower():
#             text = re.sub(re.escape(full), short, text, flags=re.IGNORECASE, count=1)
    
#     return text

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import time
import re
import asyncio
from functools import lru_cache

# Initialize FastAPI app
app = FastAPI(title="T5 Paraphrasing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Request & Response Schemas
class ParaphraseRequest(BaseModel):
    text: str
    strength: Literal["light", "medium", "strong"] = "medium"

class ParaphraseResponse(BaseModel):
    paraphrased_text: str
    model_used: str
    similarity: float
    processing_time: float

# Load models
@lru_cache(maxsize=1)
def load_paraphrase_model():
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
    return tokenizer, model

@lru_cache(maxsize=1)
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def clean_output(text):
    return re.sub(r'\s+', ' ', text).replace(" .", ".").strip()

def calculate_similarity(text1: str, text2: str) -> float:
    model = load_similarity_model()
    embeddings = model.encode([text1, text2])
    return float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))

def format_prompt(text: str) -> str:
    return f"Paraphrase the following sentence without changing its meaning or adding extra details: {text}"

async def t5_paraphrase(text: str, strength: str = "medium") -> str:
    tokenizer, model = load_paraphrase_model()
    prompt = format_prompt(text)

    encoded = tokenizer.encode_plus(prompt, return_tensors="pt", max_length=256, truncation=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}

  
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        None,
        lambda: model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=256,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=1
        )
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(result)


@app.post("/api/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(req: ParaphraseRequest):
    start_time = time.time()
    try:
        paraphrased = await t5_paraphrase(req.text, req.strength)
        similarity = calculate_similarity(req.text, paraphrased)

        return ParaphraseResponse(
            paraphrased_text=paraphrased,
            model_used="humarin/chatgpt_paraphraser_on_T5_base",
            similarity=round(similarity, 4),
            processing_time=round(time.time() - start_time, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "T5 Paraphrasing API is running"}
