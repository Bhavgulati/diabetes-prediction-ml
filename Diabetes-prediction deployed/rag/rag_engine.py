"""
DiabetesAI Lightweight RAG Engine
===================================
Uses keyword-based semantic search instead of ChromaDB.
RAM usage: ~2MB instead of ~300MB
Works perfectly on Render free tier.
"""

import os
import re

# ── Path to knowledge base ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── In-memory knowledge base ──
_knowledge_base = None

def get_knowledge_base():
    """Load knowledge base into memory — lazy, runs once"""
    global _knowledge_base
    if _knowledge_base is not None:
        return _knowledge_base
    try:
        from medical_knowledge import MEDICAL_DOCUMENTS
        _knowledge_base = MEDICAL_DOCUMENTS
        print(f"✅ Lightweight RAG loaded {len(_knowledge_base)} documents")
    except Exception as e:
        print(f"⚠️  RAG knowledge base load failed: {e}")
        _knowledge_base = []
    return _knowledge_base


def is_knowledge_base_ready():
    """Check if knowledge base is available"""
    try:
        docs = get_knowledge_base()
        return len(docs) > 0
    except Exception:
        return False


def retrieve_context(query, n_results=3, category_filter=None):
    """
    Keyword-based retrieval — finds most relevant medical documents.
    No vector DB needed. Uses word overlap scoring.
    """
    docs = get_knowledge_base()
    if not docs:
        return []

    query_lower = query.lower()
    # Extract meaningful words (remove common stop words)
    stop_words = {'the','a','an','is','are','was','were','be','been','being',
                  'have','has','had','do','does','did','will','would','could',
                  'should','may','might','shall','can','i','my','me','you','your',
                  'what','how','when','where','why','which','who','that','this',
                  'for','with','about','if','or','and','to','of','in','on','at'}
    query_words = set(re.findall(r'\b\w+\b', query_lower)) - stop_words

    scored = []
    for doc in docs:
        # Skip if category filter doesn't match
        if category_filter and doc.get('category') != category_filter:
            continue

        content_lower = doc['content'].lower()
        # Count keyword matches
        match_score = sum(1 for word in query_words if word in content_lower)

        # Bonus for exact phrase matches
        if len(query_lower) > 5 and query_lower[:20] in content_lower:
            match_score += 3

        # Bonus for title/category match
        if any(word in doc.get('category','').lower() for word in query_words):
            match_score += 2

        if match_score > 0:
            scored.append((match_score, doc))

    # Sort by score, take top n
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:n_results]]

    # Format as chunks
    results = []
    for doc in top_docs:
        # Take first 400 words of content
        words   = doc['content'].split()
        excerpt = ' '.join(words[:400])
        results.append({
            'text':     excerpt.strip(),
            'source':   doc['source'],
            'category': doc['category'],
            'score':    0.8  # placeholder score
        })

    return results


def format_context_for_claude(retrieved_chunks):
    """Format retrieved chunks into context string for Claude"""
    if not retrieved_chunks:
        return ""
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Category: {chunk['category']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def detect_query_category(query):
    """Map user question to document category for better retrieval"""
    query_lower = query.lower()

    if any(w in query_lower for w in ['eat','food','diet','meal','rice','sugar',
                                        'fruit','vegetable','drink','avoid','cook',
                                        'breakfast','lunch','dinner','snack','sweet',
                                        'ice cream','chocolate','restaurant','carb']):
        return 'diet'

    if any(w in query_lower for w in ['exercise','walk','gym','workout','run',
                                        'swim','yoga','sport','activity','physical']):
        return 'exercise'

    if any(w in query_lower for w in ['medicine','medication','metformin','insulin',
                                        'tablet','drug','prescription','dose']):
        return 'medication'

    if any(w in query_lower for w in ['stress','anxiety','depression','mental',
                                        'worry','sad','overwhelmed','burnout']):
        return 'mental_health'

    if any(w in query_lower for w in ['complication','kidney','eye','foot','nerve',
                                        'heart','vision','amputation','neuropathy']):
        return 'complications'

    if any(w in query_lower for w in ['prevent','reduce risk','pre-diabetes',
                                        'borderline','family history']):
        return 'prevention'

    if any(w in query_lower for w in ['monitor','check','test','hba1c','glucose',
                                        'blood sugar','meter','reading']):
        return 'monitoring'

    return None  # search all categories


def get_rag_stats():
    """Return stats about the knowledge base"""
    try:
        docs = get_knowledge_base()
        return {
            'status':        'ready' if docs else 'empty',
            'total_chunks':  len(docs),
            'engine':        'lightweight-keyword',
            'ram_usage':     '~2MB'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def build_knowledge_base():
    """Compatibility function — just loads docs into memory"""
    docs = get_knowledge_base()
    print(f"✅ Lightweight RAG ready: {len(docs)} documents loaded")
    return len(docs)