"""
DiabetesAI RAG Setup Script
============================
Run this ONCE to build the vector database.
After this, the knowledge base persists on disk.

Usage:
    cd "Diabetes-prediction deployed"
    python rag/setup_rag.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("  DiabetesAI RAG Knowledge Base Setup")
    print("=" * 60)
    print()

    print("Step 1: Loading medical knowledge documents...")
    try:
        from medical_knowledge import MEDICAL_DOCUMENTS
        print(f"  ✅ Loaded {len(MEDICAL_DOCUMENTS)} medical documents")
        categories = set(d['category'] for d in MEDICAL_DOCUMENTS)
        print(f"  📚 Categories: {', '.join(sorted(categories))}")
    except Exception as e:
        print(f"  ❌ Failed to load documents: {e}")
        return

    print()
    print("Step 2: Initialising ChromaDB vector database...")
    try:
        from rag_engine import get_collection, CHROMA_PATH
        col = get_collection()
        print(f"  ✅ ChromaDB initialised at: {CHROMA_PATH}")
    except Exception as e:
        print(f"  ❌ ChromaDB error: {e}")
        return

    print()
    print("Step 3: Loading embedding model (all-MiniLM-L6-v2)...")
    print("  ⏳ First run downloads ~90MB model — please wait...")
    try:
        from rag_engine import build_knowledge_base
        print("  ✅ Embedding model loaded")
    except Exception as e:
        print(f"  ❌ Embedding model error: {e}")
        return

    print()
    print("Step 4: Building vector database (chunking + embedding)...")
    try:
        total_chunks = build_knowledge_base()
        print(f"  ✅ Built {total_chunks} vector chunks")
    except Exception as e:
        print(f"  ❌ Build error: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("Step 5: Testing retrieval...")
    try:
        from rag_engine import retrieve_context, format_context_for_claude
        test_queries = [
            "What should I eat if I have diabetes?",
            "How much exercise should I do?",
            "What is HbA1c?"
        ]
        for query in test_queries:
            results = retrieve_context(query, n_results=2)
            print(f"  ✅ '{query[:40]}...' → {len(results)} chunks retrieved")
            if results:
                print(f"     Top match: {results[0]['source']} (score: {results[0]['score']})")
    except Exception as e:
        print(f"  ❌ Retrieval test failed: {e}")
        return

    print()
    print("=" * 60)
    print("  🎉 RAG Knowledge Base Ready!")
    print("=" * 60)
    print()
    print("Your chatbot now uses WHO + ADA medical guidelines.")
    print("Restart your Flask app and test the chatbot!")
    print()
    print("Example questions to try:")
    print("  - 'Can I eat ice cream?'")
    print("  - 'What exercises are best for diabetes?'")
    print("  - 'How do I manage stress with diabetes?'")
    print("  - 'What are the warning signs of complications?'")

if __name__ == '__main__':
    main()