"""
CloudSufi RAG Q&A - Core RAG Engine
Handles PDF processing, embedding, vector storage, and retrieval
Uses Claude API (Anthropic) for generation and sentence-transformers for embeddings
"""

import os
from typing import List, Dict
from pypdf import PdfReader
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import re


class RAGEngine:
    """RAG engine for document Q&A with citations using Gemini API"""

    def __init__(self, api_key: str):
        """Initialize RAG engine with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-flash-latest")

        # Initialize sentence transformer for embeddings (runs locally, no API needed)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB (in-memory)
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.chunk_size = 1000  # characters per chunk
        self.chunk_overlap = 200  # overlap between chunks

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                # Add page number marker for citation
                text += f"\n[PAGE {page_num + 1}]\n{page_text}\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")

    def chunk_text(self, text: str, source_name: str) -> List[Dict]:
        """
        Split text into chunks with overlap for better context preservation
        Returns list of dicts with chunk text, metadata
        """
        chunks = []

        # Simple sentence-aware chunking
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        current_page = 1
        chunk_id = 0

        for sentence in sentences:
            # Track page numbers
            if "[PAGE" in sentence:
                match = re.search(r"\[PAGE (\d+)\]", sentence)
                if match:
                    current_page = int(match.group(1))
                sentence = re.sub(r"\[PAGE \d+\]", "", sentence).strip()
                if not sentence:
                    continue

            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += " " + sentence
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "metadata": {
                                "source": source_name,
                                "page": current_page,
                                "chunk_id": chunk_id,
                            },
                        }
                    )
                    chunk_id += 1

                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-50:] if len(words) > 50 else words
                current_chunk = " ".join(overlap_words) + " " + sentence

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "metadata": {
                        "source": source_name,
                        "page": current_page,
                        "chunk_id": chunk_id,
                    },
                }
            )

        return chunks

    def process_documents(self, pdf_paths: List[str]) -> int:
        """
        Process PDF documents: extract text, chunk, embed, store in vector DB
        Returns number of chunks processed
        """
        # Create new collection
        collection_name = "documents"
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=collection_name, metadata={"description": "Document chunks for Q&A"}
        )

        all_chunks = []

        # Process each PDF
        for pdf_path in pdf_paths:
            source_name = os.path.basename(pdf_path)
            print(f"Processing: {source_name}")

            # Extract and chunk
            text = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text, source_name)
            all_chunks.extend(chunks)

        # Generate embeddings and add to ChromaDB
        if all_chunks:
            texts = [chunk["text"] for chunk in all_chunks]
            metadatas = [chunk["metadata"] for chunk in all_chunks]
            ids = [f"chunk_{i}" for i in range(len(all_chunks))]

            # Get embeddings using sentence-transformers (local, free!)
            print(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings = embeddings.tolist()  # Convert to list for ChromaDB

            # Add to collection
            self.collection.add(
                embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
            )

        return len(all_chunks)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if not self.collection:
            return []

        # Get query embedding using sentence-transformers
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        # Format results
        chunks = []
        if results["documents"][0]:
            for i in range(len(results["documents"][0])):
                chunks.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

        return chunks

    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a question using RAG:
        1. Retrieve relevant chunks
        2. Generate answer with Claude
        3. Include citations
        """
        # Retrieve context
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)

        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information in the documents to answer this question.",
                "citations": [],
                "context_used": [],
            }

        # Build context for Claude
        context = "\n\n".join(
            [
                f"[Source: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}]\n{chunk['text']}"
                for chunk in relevant_chunks
            ]
        )

        # System prompt for Claude
        system_prompt = """You are a helpful assistant that answers questions based on provided document excerpts.

Rules:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain the answer, say so clearly
3. Be concise but complete
4. When referencing information, mention which source and page it came from
5. Use natural language, don't just copy text verbatim"""

        # Generate answer with Gemini
        try:
            prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"
            response = self.model.generate_content(prompt)
            answer = response.text

        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        # Prepare citations
        citations = []
        for chunk in relevant_chunks:
            citations.append(
                {
                    "source": chunk["metadata"]["source"],
                    "page": chunk["metadata"]["page"],
                    "distance": round(chunk["distance"], 3),
                }
            )

        return {
            "answer": answer,
            "citations": citations,
            "context_used": [chunk["text"][:200] + "..." for chunk in relevant_chunks],
        }


# Test function
if __name__ == "__main__":
    # Simple test
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
    else:
        engine = RAGEngine(api_key)
        print("RAG Engine initialized successfully!")
