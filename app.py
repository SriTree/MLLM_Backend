from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import re
import nltk
from nltk.tokenize import sent_tokenize
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import numpy as np
import os
import psutil
import logging

# Configure environment
os.environ['HF_HOME'] = '/tmp'  # Set cache to writable directory
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism to save memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

app = FastAPI(
    title="Transcript Highlighter API",
    description="API for highlighting relevant sentences in transcripts using SentenceTransformers",
    version="1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
embedding_model = None
reranker_model = None

# --- Model Loading ---
@app.on_event("startup")
async def load_models():
    """Load models with Hugging Face Spaces optimizations"""
    global embedding_model, reranker_model
    
    try:
        logger.info("Loading models...")
        
        # Smaller default models for Hugging Face Spaces free tier
        model_name = os.environ.get("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_model = SentenceTransformer(model_name, device="cpu")
        logger.info(f"Loaded embedding model: {model_name}")
        
        reranker_name = os.environ.get("RERANKER_MODEL_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker_model = CrossEncoder(reranker_name, device="cpu")
        logger.info(f"Loaded reranker model: {reranker_name}")
        
        logger.info(f"Current memory usage: {psutil.virtual_memory().percent}%")
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# --- Request Models ---
class QuestionRequest(BaseModel):
    question: str
    context: str
    threshold: float = 0.7
    chunk_size: int = 4
    chunk_overlap: int = 2
    top_k_retrieval: int = 15
    top_k_final: int = 3
    max_highlight_sentences: int = 1
    hybrid_search_weight: float = 0.3
    answer_strictness: float = 0.8
    score_gap_threshold: float = 0.15

class HighlightResponse(BaseModel):
    answer: str
    confidence: float
    highlights: List[Dict[str, Any]]
    merged_highlights: List[Dict[str, Any]]

# --- Helper Functions --- 
def segment_sentences(text: str) -> List[str]:
    """Optimized sentence segmentation with list detection"""
    sentences = sent_tokenize(text)
    final_sentences = []
    
    for sentence in sentences:
        # Handle list patterns like "X including A, B, and C"
        if (', and ' in sentence or '; and ' in sentence) and sentence.count(',') >= 2:
            list_pattern = re.search(r'(like|including|such as)([^,.]*,.*?, and .*)', sentence)
            if list_pattern:
                intro = sentence[:list_pattern.start(2)]
                items = list_pattern.group(2).split(', and ')
                if len(items) > 1:
                    last_item = items[-1]
                    other_items = items[0].split(', ')
                    for item in other_items:
                        final_sentences.append(f"{intro} {item}")
                    final_sentences.append(f"{intro} {last_item}")
                    continue
        final_sentences.append(sentence)
    
    return [s.strip() for s in final_sentences if s.strip()]

def extract_key_terms(question: str) -> Dict[str, Any]:
    """Extract key terms with minimal processing"""
    cleaned = re.sub(r'[^\w\s]', '', question.lower())
    words = cleaned.split()
    
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "and", "or"}
    question_words = {"who", "what", "where", "when", "why", "how"}
    
    key_terms = [word for word in words if (word not in stop_words or word in question_words)]
    
    return {
        "key_terms": set(key_terms),
        "noun_phrases": set(),
        "relationship": {"subject": None, "action": None, "object": None}
    }

def create_chunks(sentences: List[str], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Create overlapping chunks with position tracking"""
    chunks = []
    for i in range(0, len(sentences), max(1, chunk_size - chunk_overlap)):
        end_idx = min(i + chunk_size, len(sentences))
        chunk_text = " ".join(sentences[i:end_idx])
        chunks.append({
            "text": chunk_text,
            "start_idx": i,
            "end_idx": end_idx - 1,
            "sentences": sentences[i:end_idx],
            "sentence_indices": list(range(i, end_idx))
        })
    return chunks

def score_relevance(question: str, sentences: List[str], extracted_info: Dict[str, Any]) -> List[tuple]:
    """Score sentences using available models"""
    if reranker_model:
        pairs = [(question, sent) for sent in sentences]
        scores = reranker_model.predict(pairs)
        return [(i, float(score)) for i, score in enumerate(scores)]
    
    # Fallback to embedding similarity
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(
        question_embedding.unsqueeze(0),
        sentence_embeddings
    ).tolist()
    
    return [(i, score) for i, score in enumerate(scores)]

def merge_adjacent_sentences(highlights: List[Dict[str, Any]], all_sentences: List[str]) -> List[Dict[str, Any]]:
    """Merge consecutive highlights"""
    if not highlights:
        return []
    
    highlights.sort(key=lambda x: x["index"])
    merged = []
    current = highlights[0].copy()
    
    for h in highlights[1:]:
        if h["index"] == current["index"] + 1:
            current["end_index"] = h["index"]
            current["score"] = max(current["score"], h["score"])
        else:
            merged.append(current)
            current = h.copy()
    
    merged.append(current)
    
    # Add merged text
    for m in merged:
        m["text"] = " ".join(all_sentences[m["index"]:m["end_index"]+1])
    
    return merged

# --- API Endpoints ---
@app.post("/api/query", response_model=HighlightResponse)
async def process_query(request: QuestionRequest):
    """Main query processing endpoint"""
    try:
        # 1. Preprocess input
        extracted_info = extract_key_terms(request.question)
        sentences = segment_sentences(request.context)
        if not sentences:
            return {
                "answer": "No content to analyze",
                "confidence": 0.0,
                "highlights": [],
                "merged_highlights": []
            }

        # 2. Create chunks
        chunks = create_chunks(sentences, request.chunk_size, request.chunk_overlap)
        
        # 3. First-stage retrieval
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            if reranker_model:
                score = reranker_model.predict([(request.question, chunk["text"])])[0]
            else:
                question_embedding = embedding_model.encode(request.question, convert_to_tensor=True)
                chunk_embedding = embedding_model.encode(chunk["text"], convert_to_tensor=True)
                score = torch.nn.functional.cosine_similarity(
                    question_embedding.unsqueeze(0),
                    chunk_embedding.unsqueeze(0)
                ).item()
            chunk_scores.append((i, float(score)))
        
        # 4. Get top candidate sentences
        top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:request.top_k_retrieval]
        candidate_sentences = []
        for idx, _ in top_chunks:
            chunk = chunks[idx]
            for sent_idx in chunk["sentence_indices"]:
                if sentences[sent_idx] not in candidate_sentences:
                    candidate_sentences.append(sentences[sent_idx])
        
        # 5. Fine-grained scoring
        scored_sentences = score_relevance(request.question, candidate_sentences, extracted_info)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 6. Apply thresholds and filters
        results = [
            (i, score) for i, score in scored_sentences 
            if score >= request.threshold
        ][:request.max_highlight_sentences]
        
        # 7. Prepare response
        highlights = []
        for i, score in results:
            try:
                original_idx = sentences.index(candidate_sentences[i])
                highlights.append({
                    "index": original_idx,
                    "text": candidate_sentences[i],
                    "score": score
                })
            except ValueError:
                continue
        
        # 8. Merge adjacent highlights
        merged = merge_adjacent_sentences(highlights, sentences)
        
        # 9. Prepare final answer
        primary_answer = highlights[0]["text"] if highlights else "No relevant content found"
        confidence = min((highlights[0]["score"] * 100 if highlights else 0), 100.0)
        
        return {
            "answer": primary_answer,
            "confidence": confidence,
            "highlights": highlights,
            "merged_highlights": merged
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """System health endpoint"""
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "memory": {
            "available": f"{mem.available / (1024**2):.1f}MB",
            "used_percent": f"{mem.percent}%"
        },
        "models": {
            "embedding": os.environ.get("EMBEDDING_MODEL_PATH", "default"),
            "reranker": os.environ.get("RERANKER_MODEL_PATH", "default"),
            "status": "loaded" if embedding_model and reranker_model else "error"
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Transcript Highlighter API",
        "docs": "/docs",
        "healthcheck": "/api/health"
    }
