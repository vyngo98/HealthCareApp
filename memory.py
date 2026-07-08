## Short-term Memory
from llama_index.core.memory import Memory

# memory = Memory.from_defaults(session_id="my_session",
#                               token_limit=40000 # Normally you would set this to be closer to the LLM context window (i.e. 75,000, etc.)
#                               )

## Long term Memory
from llama_index.core.memory import StaticMemoryBlock

# Static Information as long-term memory
static_info_block = StaticMemoryBlock(
    name="core_info",
	static_content="My name is Tuana, and I live in Amsterdam. I work at LlamaIndex.",
	priority=0
)

# Facts as long-term memory
#  The FactExtractionMemoryBlock is a unique long-term memory block that is initialized with a default prompt
#  (which you can override), that instructs an LLM to extract a list of facts from ongoing conversations.
#  To initialize this block:
from llama_index.core.memory import FactExtractionMemoryBlock
from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="qwen3:8b",
    request_timeout=600
)

facts_block = FactExtractionMemoryBlock(
    name="extracted_info",
    llm=llm,
    max_facts=50,
    priority=1
)

# Example how to get fact after a long conversation
# print(memory.memory_blocks[1].facts)

# ['User is 29 years old.', 'User has a sister.']

# Vector search as long-term memory
from llama_index.core.memory import VectorMemoryBlock
from sentence_transformers import SentenceTransformer
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model =  SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="./memory_chroma_db")
collection = db.get_or_create_collection(name="memory_store")
vector_store = ChromaVectorStore(chroma_collection=collection)

vector_block = VectorMemoryBlock(
    name="vector_memory",
    # required: pass in a vector store like qdrant, chroma, weaviate, milvus, etc.
    vector_store=vector_store,
    priority=2,
    embed_model=embed_model,
    similarity_top_k=2,
)

# Customizing Memory
from typing import Optional, List, Any
from llama_index.core.llms import ChatMessage
from llama_index.core.memory.memory import BaseMemoryBlock

class MentionCounter(BaseMemoryBlock[str]):
    """
    A memory block that counts the number of times a user mentions a specific name.
    """
    mention_name: str = "Logan"
    mention_count: int = 0

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        return f"Logan was mentioned {self.mention_count} times."

    async def _aput(self, messages: List[ChatMessage]) -> None:
        for message in messages:
            if self.mention_name in message.content:
                self.mention_count += 1

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        return ""

blocks = [static_info_block, facts_block, vector_block]

memory = Memory.from_defaults(
    session_id="my_session",
    # Setting a extremely low ratio so that more tokens are flushed to long-term memory
    chat_history_token_ratio=0.02,
    token_flush_size=500,
    memory_blocks=blocks,
    # insert into the latest user message, can also be "system"
    insert_method="system")

# With this, we can simulate a conversation with an agent and inspect the long-term memory.
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(
    tools=[],
    llm=llm,
)

user_msgs = [
    "Hi! My name is Logan",
    "What is your opinion on minature shnauzers?",
    "Do they shed a lot?",
    "What breeds are comparable in size?",
    # "What is your favorite breed?",
    # "Would you recommend owning a dog?",
    # "What should I buy to prepare for owning a dog?",
]

for user_msg in user_msgs:
    # _ = await agent.run(user_msg=user_msg, memory=memory)
    _ = agent.run(user_msg=user_msg, memory=memory)

# chat_history = await memory.aget()
chat_history = memory.aget()
print(len(chat_history))

for block in chat_history[-2].blocks:
    print(block.text)