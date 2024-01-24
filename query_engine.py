from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index import get_response_synthesizer
from document_summary_index import CHAT_TREE_SUMMARIZE_PROMPT


DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
    "The documents are listed below. Each document is numbered next to it, along with a summary of the document."
    "Please answer the number of the documents you need to reference to answer the question, in order of relevance."
    "The relevance score is a number from 1-10 based on how relevant the document seems to be to the question. \n\n"
    "Be sure to use the following format."
    "Never describe the text in any other way. \n\n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>S\nCommentary of document 2>\nCommentary of document 2>"
    "... \n<summary of document 1>\fnContent"
    "Document 10:\n<summary of document 10>\n\n"
    "Question:<question>\\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7"
    "Doc: 3, Relevance: 4"
    "Doc: 7, Relevance: 3\n"
    "So let's begin. \n\nContext_str"
    "{context_str}\n"
    "Question: {query_str}\n"
    "Answer:\n"
)

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
)

query_response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,
    summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
    verbose=True,
)
