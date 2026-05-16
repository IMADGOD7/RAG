import streamlit as st
from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Assistant")
st.markdown("Ask questions based on your knowledge base")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize models
@st.cache_resource
def load_models():
    llm = ChatMistralAI(model="mistral-small-2506")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ])
    return llm, embeddings, vectorstore, retriever, prompt

try:
    llm, embeddings, vectorstore, retriever, prompt = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    num_docs = st.slider("Number of documents to retrieve", 1, 10, 4)
    show_sources = st.checkbox("Show source documents", True)

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input("Ask a question:", placeholder="Type your question here...")

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

if user_query and search_button:
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        try:
            with st.spinner("Searching and generating response..."):
                # Retrieve documents
                docs = retriever.invoke(user_query)
                
                if not docs:
                    st.warning("No documents found in the knowledge base for your query.")
                else:
                    # Create context
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Get response
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": user_query
                    })
                    
                    response = llm.invoke(final_prompt)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_query,
                        "answer": response.content,
                        "docs_count": len(docs)
                    })
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📝 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            
            if show_sources:
                st.caption(f"📄 {chat['docs_count']} documents retrieved")
            
            st.divider()

# Clear history button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()