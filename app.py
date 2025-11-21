## RAG Q&A Conversation With PDF Including Chat History

# ---- Import required libraries ----
import streamlit as st                               # Web UI framework
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma                  # Vector database (ChromaDB)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq                  # Groq LLM integration
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings  # Text embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # Load PDF files
import os
from dotenv import load_dotenv                       # For environment variables

# ---- Load environment variables ----
load_dotenv()                                        # Read .env file
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")       # Set HuggingFace token
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Load embedding model

# ---- Set up Streamlit interface ----
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# ---- Input field for Groq API key ----
api_key = st.text_input("Enter your Groq API key:", type="password")

# ---- Only continue if user provides Groq key ----
if api_key:
    # Initialize the Groq LLM (Gemma2-9B model)
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # ---- Session and chat setup ----
    session_id = st.text_input("Session ID", value="default_session")  # Allow multiple chat sessions

    # Initialize a place to store all chat histories in Streamlit’s session state
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # ---- File uploader for PDFs ----
    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

    # ---- Process uploaded PDFs ----
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"                   # Temporary file path
            with open(temppdf, "wb") as file:        # Save uploaded file locally
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name        # Store original filename (optional)

            # Load text from PDF pages
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)                    # Add pages to the main list

        # ---- Split text and create embeddings ----
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)               # Split into smaller chunks
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)  # Store in Chroma
        retriever = vectorstore.as_retriever()                          # Make a retriever for search

        # ---- Contextualization prompt for follow-up questions ----
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        # Define how messages are passed into the model for contextualization
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),  # Instruction
                MessagesPlaceholder("chat_history"),         # Previous messages
                ("human", "{input}"),                        # Latest user question
            ]
        )

        # Create a history-aware retriever (reformulates user query using context)
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # ---- Define QA system prompt ----
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"  # Placeholder for retrieved text
        )

        # Build structured QA prompt (includes history + question)
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create document-combining QA chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Combine retriever + QA chain into one RAG pipeline
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # ---- Manage chat history per session ----
        def get_session_history(session: str) -> BaseChatMessageHistory:
            """Return the stored chat history for the session (or create new one)."""
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Wrap the RAG chain so it remembers chat history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # ---- User question input ----
        user_input = st.text_input("Your question:")

        # ---- Run RAG pipeline when user submits a question ----
        if user_input:
            session_history = get_session_history(session_id)   # Get conversation memory
            response = conversational_rag_chain.invoke(         # Run the full RAG chain
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}  # Keeps messages separated per session
                },
            )

            # ---- Display outputs ----
            st.write(st.session_state.store)                    # Debug view of stored sessions
            st.write("**Assistant:**", response['answer'])      # Show model’s answer
            st.write("**Chat History:**", session_history.messages)  # Show conversation log

# ---- If no API key provided ----
else:
    st.warning("Please enter the Groq API Key")