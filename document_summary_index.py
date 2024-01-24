from llama_index import ServiceContext, get_response_synthesizer
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import ChatMessage, MessageRole
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.prompts.base import ChatPromptTemplate
from llama_index.llms import OpenAI
from llama_index import LLMPredictor
import nest_asyncio

nest_asyncio.apply()

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are the world's trusted QA system. \n"
        "Always answer queries using the contextual information provided, not prior knowledge. \n"
        "Some rules to follow:\n"
        "1. do not refer directly to the context specified in the answer. \n"
        "2. 'based on the context,...' or 'Context information is...' or similar statements should be avoided."
    ),
    role=MessageRole.SYSTEM,
)

TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information from multiple sources is shown below. \n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Answer the question considering information from multiple sources, not prior knowledge. \n"
            "If in doubt, answer 'no information'. \n"
            "Query: {query_str}\n"
            "Answer:"
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)

SUMMARY_QUERY = "Summarize the content of the text provided."

llm = OpenAI(temperature=0.1, model="gpt-4")
llama_debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug_handler])

service_context = ServiceContext.from_defaults(
    llm=llm,
    callback_manager=callback_manager,
    chunk_size=1024,
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,
    text_qa_template=TEXT_QA_SYSTEM_PROMPT,
    summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
    verbose=True,
)


def getDocumentSummaryIndex(document):
    return DocumentSummaryIndex.from_documents(
        document,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        summary_query=SUMMARY_QUERY,
    )
