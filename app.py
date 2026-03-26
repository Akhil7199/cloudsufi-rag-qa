"""
CloudSufi RAG Q&A - Streamlit UI
Simple, clean interface for document Q&A
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import RAGEngine
import tempfile

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="CloudSufi RAG Q&A", page_icon="📄", layout="wide")

# Initialize session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def main():
    # Title
    st.title("📄 Document Q&A System")
    st.markdown("**RAG-based Question Answering with Citations**")
    st.markdown("---")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Upload Documents")

        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Enter your Gemini API key or set it in .env file",
        )

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF documents (1-3 files)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload 1 to 3 PDF documents",
        )

        # Process button
        if st.button("🚀 Process Documents", type="primary"):
            if not api_key:
                st.error("Please provide a Gemini API key")
            elif not uploaded_files:
                st.error("Please upload at least one PDF document")
            elif len(uploaded_files) > 3:
                st.error("Please upload maximum 3 documents")
            else:
                process_documents(api_key, uploaded_files)

        # Status
        if st.session_state.documents_processed:
            st.success("✅ Documents ready for Q&A!")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses:
        - **RAG** (Retrieval Augmented Generation)
        - **Gemini 1.5 Flash** for answers
        - **ChromaDB** for vector storage
        - **sentence-transformers** for embeddings
        """)

    # Main area for Q&A
    if st.session_state.documents_processed:
        show_qa_interface()
    else:
        show_welcome_message()


def process_documents(api_key: str, uploaded_files):
    """Process uploaded PDF documents"""
    with st.spinner("Processing documents... This may take a minute."):
        try:
            # Save uploaded files temporarily
            temp_paths = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_paths.append(tmp_file.name)

            # Initialize RAG engine
            st.session_state.rag_engine = RAGEngine(api_key)

            # Process documents
            num_chunks = st.session_state.rag_engine.process_documents(temp_paths)

            # Clean up temp files
            for path in temp_paths:
                try:
                    os.unlink(path)
                except:
                    pass

            st.session_state.documents_processed = True
            st.sidebar.success(
                f"✅ Processed {len(uploaded_files)} documents ({num_chunks} chunks)"
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")


def show_welcome_message():
    """Show welcome message when no documents are loaded"""
    st.info("👈 Please upload PDF documents using the sidebar to get started")

    st.markdown("### How to use:")
    st.markdown("""
    1. **Enter your Gemini API key** in the sidebar (or set it in `.env` file)
    2. **Upload 1-3 PDF documents** you want to ask questions about
    3. **Click "Process Documents"** to prepare them for Q&A
    4. **Ask questions** in natural language
    5. **Get answers with citations** showing which document and page the information came from
    """)

    st.markdown("### Example questions you can ask:")
    st.markdown("""
    - "What is the main topic of the document?"
    - "What are the key findings mentioned?"
    - "Can you summarize the methodology described?"
    - "What recommendations are provided?"
    """)


def show_qa_interface():
    """Show Q&A interface when documents are loaded"""

    # Chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("citations"):
                with st.expander("📚 View Citations"):
                    for i, citation in enumerate(chat["citations"], 1):
                        st.markdown(f"""
                        **Citation {i}:**
                        - Source: `{citation["source"]}`
                        - Page: {citation["page"]}
                        - Distance: {citation["distance"]:.3f} (lower is better)
                        """)

    # Question input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Add user message
        with st.chat_message("user"):
            st.write(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_engine.answer_question(question)

                # Display answer
                st.write(result["answer"])

                # Display citations
                if result["citations"]:
                    with st.expander("📚 View Citations"):
                        for i, citation in enumerate(result["citations"], 1):
                            st.markdown(f"""
                            **Citation {i}:**
                            - Source: `{citation["source"]}`
                            - Page: {citation["page"]}
                            - Distance: {citation["distance"]:.3f} (lower is better)
                            """)

                # Save to history
                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "citations": result["citations"],
                    }
                )


if __name__ == "__main__":
    main()
