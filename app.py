import os
import streamlit as st

from dotenv import load_dotenv
from rag_backend import RAGChatbot


# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="DocuMind | AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


# =========================================
# 2. CUSTOM CSS
# =========================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #f7f7fb;
    color: #202033;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e9e8f0;
}

[data-testid="stSidebar"] h1 {
    color: #272342;
    font-size: 25px;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #393653;
}

/* Header */
.main-heading {
    font-size: 36px;
    font-weight: 750;
    color: #292542;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

.main-subheading {
    color: #77758b;
    font-size: 15px;
    margin-bottom: 28px;
}

/* Welcome section */
.welcome-card {
    background: linear-gradient(135deg, #ffffff, #f0edff);
    border: 1px solid #e7e1ff;
    border-radius: 18px;
    padding: 30px;
    margin: 20px 0;
}

.welcome-card h2 {
    color: #302951;
    font-size: 24px;
    margin-bottom: 10px;
}

.welcome-card p {
    color: #77738f;
    font-size: 15px;
}

/* Feature cards */
.feature-card {
    background-color: #ffffff;
    border: 1px solid #e9e7f2;
    border-radius: 14px;
    padding: 18px;
    min-height: 115px;
}

.feature-card h4 {
    color: #393253;
    margin-bottom: 7px;
    font-size: 15px;
}

.feature-card p {
    color: #858198;
    font-size: 13px;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: none;
    background-color: #7057d9;
    color: white;
    font-weight: 600;
    padding: 10px 15px;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #5941bd;
    color: white;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #ffffff;
    border: 1px solid #eceaf3;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 12px;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 14px;
}

/* Document status */
.status-card {
    background-color: #f4f1ff;
    border: 1px solid #e4ddff;
    border-radius: 12px;
    padding: 14px;
    color: #51418d;
    margin-top: 10px;
}

/* Evidence */
[data-testid="stExpander"] {
    background-color: #faf9ff;
    border: 1px solid #e8e4f4;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #e9e7f0;
}

</style>
""", unsafe_allow_html=True)


# =========================================
# 3. SESSION STATE
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# =========================================
# 4. INITIALIZE BACKEND
# =========================================

if not api_key:
    st.error(
        "Google API key not found. "
        "Please add GOOGLE_API_KEY to your .env file."
    )
    st.stop()


@st.cache_resource
def load_chatbot(key):
    return RAGChatbot(key)


try:
    if st.session_state.chatbot is None:
        st.session_state.chatbot = load_chatbot(api_key)

except Exception as e:
    st.error(f"Could not initialize the chatbot: {e}")
    st.stop()


chatbot = st.session_state.chatbot


# =========================================
# 5. SIDEBAR - DOCUMENT WORKSPACE
# =========================================

with st.sidebar:

    st.title("📄 DocuMind")
    st.caption("Your personal AI document assistant")

    st.divider()

    st.subheader("📤 Add a document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Choose a PDF document to start asking questions."
    )

    if st.button("✨ Process Document", use_container_width=True):

        if uploaded_file is None:
            st.warning("Please select a PDF file first.")

        else:
            try:
                with st.spinner("Analyzing your document..."):

                    total_chunks = chatbot.process_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name
                    )

                    st.session_state.uploaded_name = uploaded_file.name
                    st.session_state.total_chunks = total_chunks
                    st.session_state.messages = []

                st.success("Document is ready!")
                st.rerun()

            except Exception as e:
                st.error(f"Processing failed: {e}")

    st.divider()

    st.subheader("📚 Current document")

    if st.session_state.uploaded_name:

        st.markdown(
            f"""
            <div class="status-card">
                <strong>🟢 Document ready</strong><br><br>
                📄 {st.session_state.uploaded_name}<br>
                <small>{st.session_state.total_chunks} text chunks indexed</small>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button("🗑️ Remove Document", use_container_width=True):

            try:
                chatbot.clear_documents()

                st.session_state.uploaded_name = None
                st.session_state.total_chunks = 0
                st.session_state.messages = []

                st.success("Document removed.")
                st.rerun()

            except Exception as e:
                st.error(f"Could not remove document: {e}")

    else:
        st.info("No document uploaded yet.")

    st.divider()

    st.subheader("💬 Conversation")

    st.caption(
        f"{len(st.session_state.messages) // 2} questions asked"
    )

    if st.button("Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Powered by Gemini AI and ChromaDB")


# =========================================
# 6. MAIN HEADER
# =========================================

st.markdown(
    '<div class="main-heading">Your documents, understood.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subheading">'
    'Upload a PDF, ask questions, and get answers grounded in your document.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================
# 7. DOCUMENT OVERVIEW
# =========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📄 Smart Document Reading</h4>
        <p>Extract and understand information from your PDF.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🔍 Contextual Search</h4>
        <p>Find relevant sections using semantic search.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 AI-Powered Answers</h4>
        <p>Get answers supported by retrieved document evidence.</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================
# 8. WELCOME SCREEN
# =========================================

if not st.session_state.messages:

    if st.session_state.uploaded_name:

        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Your document is ready!</h2>
            <p>
                Ask anything about your uploaded PDF.
                I'll search the document and provide an answer
                with supporting evidence.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to DocuMind</h2>
            <p>
                Turn lengthy documents into meaningful conversations.
                Upload a PDF using the sidebar to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Try asking questions like:")

        example_col1, example_col2 = st.columns(2)

        with example_col1:
            st.markdown("""
            - What is this document about?
            - Summarize the main points.
            """)

        with example_col2:
            st.markdown("""
            - Explain the important concepts.
            - What are the key conclusions?
            """)


# =========================================
# 9. DISPLAY CHAT HISTORY
# =========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            sources = message.get("sources", [])

            if sources:

                with st.expander(
                    f"📖 View supporting evidence ({len(sources)} chunks)"
                ):

                    for i, source in enumerate(sources, start=1):

                        st.markdown(
                            f"**Source {i} · Page {source['page']}**"
                        )

                        st.write(source["content"])

                        st.divider()


# =========================================
# 10. CHAT INPUT AND AI RESPONSE
# =========================================

question = st.chat_input(
    "Ask a question about your document..."
)

if question:

    if not st.session_state.uploaded_name:

        st.warning(
            "Please upload and process a PDF before asking questions."
        )

    else:

        # Save and display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate assistant response
        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:

                    answer, evidence = chatbot.answer_question(question)

                    st.markdown(answer)

                    if evidence:

                        with st.expander(
                            f"📖 Supporting evidence ({len(evidence)} chunks)"
                        ):

                            for i, source in enumerate(
                                evidence,
                                start=1
                            ):

                                st.markdown(
                                    f"**Source {i} · Page {source['page']}**"
                                )

                                st.write(source["content"])

                                st.divider()

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": evidence
                    })

                except Exception as e:

                    error_message = f"Something went wrong: {e}"

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })