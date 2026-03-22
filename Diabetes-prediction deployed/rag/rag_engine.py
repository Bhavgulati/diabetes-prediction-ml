"""
DiabetesAI RAG Engine
=====================
Retrieval Augmented Generation using ChromaDB + Sentence Transformers

How it works:
1. Medical documents → split into chunks → converted to vectors → stored in ChromaDB
2. User question → converted to vector → search ChromaDB → find top 3 similar chunks
3. Top chunks → passed to Claude as context → grounded medical answer
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ── Paths ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, 'chroma_db')

# ── Embedding model (free, runs locally) ──
# all-MiniLM-L6-v2: fast, small, excellent for medical Q&A
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# ── Singleton client ──
_client     = None
_collection = None

def get_collection():
    """Get or create ChromaDB collection — singleton pattern"""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    _collection = _client.get_or_create_collection(
        name='diabetes_medical_knowledge',
        embedding_function=embedding_fn,
        metadata={'hnsw:space': 'cosine'}  # cosine similarity for semantic search
    )

    return _collection


def is_knowledge_base_ready():
    """Check if the vector database has documents"""
    try:
        col = get_collection()
        return col.count() > 0
    except Exception:
        return False


def build_knowledge_base():
    """
    Build the vector database from medical documents.
    Run this ONCE — persists to disk automatically.
    """
    from medical_knowledge import MEDICAL_DOCUMENTS

    col = get_collection()

    # Clear existing data
    existing = col.get()
    if existing['ids']:
        col.delete(ids=existing['ids'])
        print(f"Cleared {len(existing['ids'])} existing chunks")

    ids        = []
    documents  = []
    metadatas  = []

    for doc in MEDICAL_DOCUMENTS:
        # Split long documents into smaller chunks for better retrieval
        chunks = chunk_text(doc['content'], chunk_size=300, overlap=50)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk.strip())
            metadatas.append({
                'source':   doc['source'],
                'category': doc['category'],
                'doc_id':   doc['id'],
                'chunk':    i
            })

    # Add to ChromaDB — embeddings generated automatically
    col.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"✅ Knowledge base built: {len(ids)} chunks from {len(MEDICAL_DOCUMENTS)} documents")
    return len(ids)


def chunk_text(text, chunk_size=300, overlap=50):
    """
    Split text into overlapping chunks.
    Overlap ensures context isn't lost at chunk boundaries.

    Example:
    chunk_size=300 words, overlap=50 words
    chunk1: words 0-300
    chunk2: words 250-550  (50 word overlap with chunk1)
    chunk3: words 500-800  (50 word overlap with chunk2)
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if len(chunk.strip()) > 30:  # skip tiny chunks
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks if chunks else [text]


def retrieve_context(query, n_results=3, category_filter=None):
    """
    Main RAG retrieval function.

    Args:
        query: User's question
        n_results: Number of relevant chunks to retrieve
        category_filter: Optional filter by category (diet, exercise, medication, etc.)

    Returns:
        List of relevant text chunks with source information
    """
    if not is_knowledge_base_ready():
        return []

    col = get_collection()

    # Build filter if category specified
    where = {'category': category_filter} if category_filter else None

    try:
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, col.count()),
            where=where,
            include=['documents', 'metadatas', 'distances']
        )

        retrieved = []
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            # cosine distance: 0 = identical, 2 = opposite
            # only include if reasonably similar (distance < 1.2)
            if distance < 1.2:
                retrieved.append({
                    'text':     results['documents'][0][i],
                    'source':   results['metadatas'][0][i]['source'],
                    'category': results['metadatas'][0][i]['category'],
                    'score':    round(1 - distance/2, 3)  # similarity 0-1
                })

        return retrieved

    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []


def format_context_for_claude(retrieved_chunks):
    """
    Format retrieved chunks into a clean context string for Claude.

    Example output:
    [Source: ADA Nutrition Therapy 2024 | Category: diet]
    Patients with diabetes should choose whole grain rice...

    [Source: WHO Guidelines 2023 | Category: lifestyle]
    Weight loss of 5-10% significantly improves blood sugar...
    """
    if not retrieved_chunks:
        return ""

    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Category: {chunk['category']} | Relevance: {chunk['score']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def detect_query_category(query):
    """
    Simple keyword-based category detection to improve retrieval.
    Maps user questions to document categories for filtered search.
    """
    query_lower = query.lower()

    if any(w in query_lower for w in ['eat', 'food', 'diet', 'meal', 'rice', 'sugar',
                                        'fruit', 'vegetable', 'drink', 'avoid', 'cook',
                                        'breakfast', 'lunch', 'dinner', 'snack', 'sweet',
                                        'ice cream', 'chocolate', 'restaurant']):
        return 'diet'

    if any(w in query_lower for w in ['exercise', 'walk', 'gym', 'workout', 'run',
                                        'swim', 'yoga', 'sport', 'activity', 'physical']):
        return 'exercise'

    if any(w in query_lower for w in ['medicine', 'medication', 'metformin', 'insulin',
                                        'tablet', 'drug', 'prescription', 'dose']):
        return 'medication'

    if any(w in query_lower for w in ['stress', 'anxiety', 'depression', 'mental',
                                        'worry', 'sad', 'overwhelmed', 'burnout']):
        return 'mental_health'

    if any(w in query_lower for w in ['complication', 'kidney', 'eye', 'foot', 'nerve',
                                        'heart', 'vision', 'amputation', 'neuropathy']):
        return 'complications'

    if any(w in query_lower for w in ['prevent', 'reduce risk', 'pre-diabetes',
                                        'borderline', 'family history']):
        return 'prevention'

    if any(w in query_lower for w in ['monitor', 'check', 'test', 'hba1c', 'glucose',
                                        'blood sugar', 'meter', 'reading']):
        return 'monitoring'

    return None  


def get_rag_stats():
    """Return stats about the knowledge base — useful for /api/health endpoint"""
    try:
        col   = get_collection()
        count = col.count()
        return {
            'status':         'ready' if count > 0 else 'empty',
            'total_chunks':   count,
            'embedding_model': EMBEDDING_MODEL,
            'vector_db':      'ChromaDB'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}