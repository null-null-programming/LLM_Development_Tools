from llama_index import ServiceContext, get_response_synthesizer
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index import VectorStoreIndex
from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import nest_asyncio
import chromadb

nest_asyncio.apply()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        """
        You are the world's trusted QA system. Answer queries using only the contextual information provided. Avoid using prior knowledge.
        Adhere to these rules:
        1. Do not refer directly to the context specified in the answer.
        2. Avoid phrases like 'based on the context...' or 'Context information is...'. 
        """
    ),
    role=MessageRole.SYSTEM,
)


TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            """
            Context information from multiple sources is provided below.
            ---------------------
            {context_str}
            ---------------------
            Answer the question using information from these sources, not prior knowledge.
            If uncertain, respond with 'no information'.
            Query: {query_str}
            Answer:
            """
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)

SUMMARY_QUERY = "Summarize the content of the text provided."

llm = OpenAI(temperature=0.1, model="gpt-4")

service_context = ServiceContext.from_defaults(llm=llm)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,
    text_qa_template=TEXT_QA_SYSTEM_PROMPT,
    summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
)


def save_index(documents):
    DocumentSummaryIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
        response_synthesizer=response_synthesizer,
        summary_query=SUMMARY_QUERY,
    )


def getVectorStoreIndex():
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )
