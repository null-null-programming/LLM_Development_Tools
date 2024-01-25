from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index import get_response_synthesizer
from document_summary_index import CHAT_TREE_SUMMARIZE_PROMPT
import nest_asyncio

nest_asyncio.apply()

DEFAULT_CHOICE_SELECT_PROMPT_TMPL = """
                                    The documents are listed below. Each document is numbered next to it, along with a summary. Please answer the number of the documents you need to reference to answer the question, in order of relevance. The relevance score is a number from 1-10 based on how relevant the document seems to be to the question. 

                                    Be sure to use the following format and never describe the text in any other way. 
                                    If you could not find actual information, please say 'no information'.

                                    Document 1:
                                    [summary of document 1]

                                    Document 2:
                                    [summary of document 2]

                                    ...

                                    Document 10:
                                    [summary of document 10]

                                    Question:
                                    [question]

                                    Answer:
                                    Doc: 9, Relevance: 7
                                    Doc: 3, Relevance: 4
                                    Doc: 7, Relevance: 3

                                    So let's begin.

                                    Context:
                                    [context_str]

                                    Question:
                                    [query_str]

                                    Answer:
                                    """

DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
    DEFAULT_CHOICE_SELECT_PROMPT_TMPL,
    prompt_type=PromptType.CHOICE_SELECT,
)

query_response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    use_async=True,
    summary_template=CHAT_TREE_SUMMARIZE_PROMPT,  # DEFAULT_CHOICE_SELECT_PROMPT,
)
