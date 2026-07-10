# Vector search as long-term memory
from llama_index.core.memory import VectorMemoryBlock
from sentence_transformers import SentenceTransformer
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import Memory

embed_model =  SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="./memory_chroma_db")
collection = db.get_or_create_collection(name="memory_store")
vector_store = ChromaVectorStore(chroma_collection=collection)

vector_block = VectorMemoryBlock(
    name="sleep_history",
    # required: pass in a vector store like qdrant, chroma, weaviate, milvus, etc.
    vector_store=vector_store,
    priority=2,
    embed_model=embed_model,
    similarity_top_k=2,
)

memory = Memory.from_defaults(
    session_id="sleep_session",
    # Setting a extremely low ratio so that more tokens are flushed to long-term memory
    chat_history_token_ratio=0.02,
    token_flush_size=500,
    memory_blocks=[vector_block],
)

