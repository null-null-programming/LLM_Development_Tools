import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
from llamaIndex import LlamaIndex


class Completions:
    """
    A class to interact with OpenAI's chat completion models and manage conversation history.

    This class handles the communication with OpenAI's API for generating chat completions
    and stores conversation histories in a MongoDB database.

    Attributes:
        client (OpenAI): An instance of the OpenAI client used for API interactions.
        messages (list): A list that stores the history of the conversation.
        mongo_client (MongoClient): A client connection to the MongoDB server.
        db (Database): A reference to a specific MongoDB database.
        conversations_collection (Collection): A MongoDB collection for storing conversation data.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of the Completions class.

        This method sets up the MongoDB client, loads environment variables, reads initial instructions
        from a JSON file, and initializes the OpenAI client for API interactions.
        """
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        self.db = self.mongo_client["llm_db"]
        self.conversations_collection = self.db["conversations"]

        load_dotenv()

        with open(".instructions.json", "r", encoding="utf-8") as json_file:
            instructions = json.load(json_file)["instructions"]

        self.messages = [{"role": "system", "content": instructions}]

        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
        )

        self.llamaIndex = LlamaIndex()

    def get_message(self, message: str):
        """
        Sends a message to the OpenAI API, stores the response, and returns it.

        Parameters:
            message (str): The user's input message to be sent to the API.
            isJson (bool): A flag indicating whether the response should be in JSON format.

        Returns:
            str: The response message from the OpenAI completion model.

        Raises:
            TypeError: If the response from the API is not a string.
        """
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=self.messages,
            response_format={"type": "json_object"},
        )

        json_data = json.loads(response.choices[0].message.content)

        if "summary" in json_data:
            return json_data

        got_message = json_data["content"]
        query_text = json_data["query_text"]

        if got_message is None:
            raise TypeError("Received message is not a string")

        searched_info = self.llamaIndex.query(query_text)

        content = f"\nLlamaIndex Info : \n{searched_info}\n"
        print(content)

        if searched_info != "No information.":
            send_message = f"{content}\n{got_message}"
            self.messages.append({"role": "assistant", "content": send_message})
        else:
            self.messages.append({"role": "assistant", "content": got_message})

        return got_message

    def get_user_input(self) -> str:
        """
        Retrieves user input from the command line.

        Returns:
            str: The user input as a string.
        """
        return input("You: ")

    def save_conversation_to_mongo(self, conversation_summary):
        """
        Saves the conversation summary to the MongoDB database.

        Parameters:
            conversation_summary (dict): The summary of the conversation in JSON format.
        """
        self.conversations_collection.insert_one(conversation_summary)
        print("Conversation saved to MongoDB.\n")

        answer = input("Do you want to update the Index to ChromaDB? (y/n) : ")
        print("")

        if (answer == "y") or (answer == "Y") or (answer == "yes") or (answer == "Yes"):
            self.save_index()

    def save_index(self):
        """
        Saves the index to the MongoDB database.
        """
        print("Start to save VectorIndex to ChromaDB.\n")
        self.llamaIndex.save()
        print("\nSaved VectorIndex to ChromaDB.\n")

    def chat(self) -> None:
        """
        Initiates and manages a chat interaction with the OpenAI chat model.

        This method facilitates a continuous conversation loop, handling user input,
        generating responses using the OpenAI API, and providing options to save or clear
        the conversation history. The loop continues until the user decides to exit.
        """
        while True:
            message = self.get_user_input()

            if message == "exit":
                self.client.close()
                break

            if message == "clear":
                self.messages = []
                continue

            if message == "save":
                summary_instruction = """
                Summarize the conversation so far in the following json format.

                {"title":str,"summary":str}
                """

                summary = self.get_message(summary_instruction)
                print(f"Summary : \n\n{summary['summary']}\n")

                self.save_conversation_to_mongo(summary)
                self.client.close()
                break

            if message == "update":
                self.save_index()
                continue

            response = self.get_message(message)
            print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    Completions().chat()
