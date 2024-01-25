import os
from dotenv import load_dotenv
from llama_index import download_loader
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from document_summary_index import getVectorStoreIndex, save_index
from query_engine import DEFAULT_CHOICE_SELECT_PROMPT, query_response_synthesizer


llm = OpenAI(temperature=0.1, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)


class LlamaIndex:
    def __init__(self):
        load_dotenv()
        self.host = os.getenv("HOST")
        self.port = int(os.getenv("PORT"))
        self.uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.query_dict = {}
        self.field_names = ["title", "summary"]
        self.SimpleMongoReader = download_loader("SimpleMongoReader")
        self.reader = self.SimpleMongoReader(self.host, self.port, self.uri)
        self.documents = self.reader.load_data(
            self.db_name,
            self.collection_name,
            self.field_names,
            query_dict=self.query_dict,
        )
        self.index = getVectorStoreIndex()
        self.query_engine = self.index.as_query_engine(
            choice_select_prompt=DEFAULT_CHOICE_SELECT_PROMPT,
            response_synthesizer=query_response_synthesizer,
        )

    def save(self):
        save_index(self.documents)

    def query(self, query_text):
        response = self.query_engine.query(query_text)
        return response


if __name__ == "__main__":
    LlamaIndex().query("What is happy?")
