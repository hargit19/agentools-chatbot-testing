import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ CORS Configuration (Fixes OPTIONS Request Issues)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ✅ API Keys (Replace with Environment Variables in Production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Connect to Pinecone Index
INDEX_NAME = "ai-agents-index"
if INDEX_NAME in pc.list_indexes().names():
    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: {INDEX_NAME}")
else:
    raise ValueError(f"Index {INDEX_NAME} not found!")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

session_memories = {}

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

class DocumentInput(BaseModel):
    source: str
    text: str

class ChatInput(BaseModel):
    session_id: str
    message: str

@app.post("/store_documents")
def store_documents(documents: list[DocumentInput]):
    """Store documents in Pinecone"""
    if not documents:
        raise HTTPException(status_code=400, detail="No documents to store")

    langchain_docs = [
        Document(page_content=doc.text, metadata={"source": doc.source})
        for doc in documents
    ]

    vectorstore.add_documents(langchain_docs)
    return {"message": f"✅ Stored {len(langchain_docs)} documents in Pinecone"}


@app.post("/chat")
def chat_with_bot(input_data: ChatInput):
    """Chat with the AI Assistant with memory"""
    session_id = input_data.session_id.strip()
    user_input = input_data.message.strip()

    if not session_id or not user_input:
        raise HTTPException(status_code=400, detail="Session ID and message cannot be empty")

    # ✅ Retrieve memory for the session (or create new)
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    memory = session_memories[session_id]

    # ✅ Create Conversational Retrieval Chain
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=vectorstore.as_retriever(), memory=memory
    )

    # ✅ Get response
    response = chat_chain.invoke({"question": user_input})
    answer = response["answer"]

    # ✅ Add messages to memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(answer)

    return {"session_id": session_id, "user": user_input, "bot": answer}










