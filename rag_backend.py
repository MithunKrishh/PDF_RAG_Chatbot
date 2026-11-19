
import os
import time
import uuid

import chromadb
from pypdf import PdfReader
from io import BytesIO

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


class RAGChatbot:

    # =========================================
    # 1. INITIALIZATION
    # =========================================

    def __init__(
        self,
        api_key,
        persist_directory="./chroma_db",
        collection_name="pdf_documents"
    ):

        print("\nInitializing RAG chatbot...", flush=True)

        # Load embedding model
        start = time.time()

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print(
            f"Embedding model loaded in {time.time() - start:.2f}s",
            flush=True
        )

        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=1024,
            max_retries=1
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.chunk_size = 800
        self.chunk_overlap = 120

        print("RAG chatbot initialized successfully.\n", flush=True)


    # =========================================
    # 2. TEXT CHUNKING
    # =========================================

    def split_text(self, text):

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + self.chunk_size,
                text_length
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - self.chunk_overlap

        return chunks


    # =========================================
    # 3. CLEAR EXISTING DOCUMENTS
    # =========================================

    def clear_documents(self):

        try:

            existing = self.collection.get()

            ids = existing.get("ids", [])

            if ids:
                self.collection.delete(ids=ids)

            print("Existing documents cleared.", flush=True)

            return True

        except Exception as e:

            print(f"Error clearing documents: {e}", flush=True)

            return False


    # =========================================
    # 4. PDF PROCESSING
    # =========================================

    def process_pdf(self, pdf_bytes, filename):

        print("\n========== PDF PROCESSING ==========", flush=True)

        start_time = time.time()

        # Step 1: Read PDF
        print("1. Reading PDF...", flush=True)

        reader = PdfReader(BytesIO(pdf_bytes))

        all_chunks = []
        all_metadata = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            page_chunks = self.split_text(text)

            for chunk in page_chunks:

                all_chunks.append(chunk)

                all_metadata.append({
                    "source": filename,
                    "page": page_number
                })

        if not all_chunks:

            raise ValueError(
                "No readable text found in the PDF. "
                "The PDF may contain scanned images."
            )

        print(
            f"2. Extracted {len(all_chunks)} chunks.",
            flush=True
        )

        # Step 2: Generate embeddings
        print("3. Generating embeddings...", flush=True)

        embedding_start = time.time()

        embeddings = self.embedding_model.encode(
            all_chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        print(
            f"Embedding time: {time.time() - embedding_start:.2f}s",
            flush=True
        )

        # Step 3: Replace old document data
        print("4. Storing in ChromaDB...", flush=True)

        self.clear_documents()

        ids = [
            str(uuid.uuid4())
            for _ in all_chunks
        ]

        self.collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadata
        )

        total_time = time.time() - start_time

        print(
            f"5. PDF processing completed in {total_time:.2f}s",
            flush=True
        )

        print("====================================\n", flush=True)

        return len(all_chunks)


    # =========================================
    # 5. RETRIEVE RELEVANT DOCUMENT CHUNKS
    # =========================================

    def retrieve_documents(self, question, k=4):

        print("Starting retrieval...", flush=True)

        start_time = time.time()

        total_documents = self.collection.count()

        if total_documents == 0:
            return []

        k = min(k, total_documents)

        # Generate question embedding
        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        evidence = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for content, metadata in zip(documents, metadatas):

            evidence.append({
                "content": content,
                "page": metadata.get("page", "Unknown"),
                "source": metadata.get("source", "Unknown")
            })

        print(
            f"Retrieval completed in {time.time() - start_time:.2f}s",
            flush=True
        )

        return evidence


    # =========================================
    # 6. EXTRACT GEMINI RESPONSE
    # =========================================

    def extract_response_text(self, response):

        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):
                    text_parts.append(item)

                elif isinstance(item, dict):

                    if item.get("type") == "text":
                        text_parts.append(
                            item.get("text", "")
                        )

            return "\n".join(text_parts)

        return str(content)


    # =========================================
    # 7. ANSWER USER QUESTION
    # =========================================

    def answer_question(self, question, k=4):

        print("\n========== RAG DEBUG ==========", flush=True)

        total_start = time.time()

        if not question or not question.strip():

            raise ValueError("Question cannot be empty.")

        print("1. Starting document retrieval...", flush=True)

        retrieval_start = time.time()

        evidence = self.retrieve_documents(
            question,
            k=k
        )

        print(
            f"2. Retrieval finished in "
            f"{time.time() - retrieval_start:.2f}s",
            flush=True
        )

        if not evidence:

            return (
                "I couldn't find any relevant information "
                "in the uploaded document.",
                []
            )

        # Prepare context
        context = "\n\n".join(
            [
                f"Source: {item['source']}, "
                f"Page: {item['page']}\n"
                f"{item['content']}"
                for item in evidence
            ]
        )

        # Build prompt
        prompt = f"""
You are a helpful AI document assistant.

Answer the user's question using the document context below.

Instructions:
1. Use the provided context as your primary source.
2. Give a clear and accurate answer.
3. Do not invent information that is not in the document.
4. If the answer cannot be found in the context,
   clearly say that the document does not provide the answer.
5. Use simple language and organize the answer properly.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        # Generate Gemini response
        print("3. Sending request to Gemini...", flush=True)

        gemini_start = time.time()

        response = self.llm.invoke(prompt)

        print(
            f"4. Gemini completed in "
            f"{time.time() - gemini_start:.2f}s",
            flush=True
        )

        # Extract answer
        answer = self.extract_response_text(response)

        print(
            f"5. Total answer time: "
            f"{time.time() - total_start:.2f}s",
            flush=True
        )

        print("===============================\n", flush=True)

        return answer, evidence
