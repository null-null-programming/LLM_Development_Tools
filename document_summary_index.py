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
# llama_debug_handler = LlamaDebugHandler()
# callback_manager = CallbackManager([llama_debug_handler])

service_context = ServiceContext.from_defaults(
    llm=llm  # , callback_manager=callback_manager
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,
    text_qa_template=TEXT_QA_SYSTEM_PROMPT,
    summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
    # verbose=True,
)


def getDocumentSummaryIndex(document):
    return DocumentSummaryIndex.from_documents(
        document,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        summary_query=SUMMARY_QUERY,
    )
