import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient


class Completions:
    """
    A class to handle conversations with OpenAI's chat completion models.

    Attributes:
        client (OpenAI): The OpenAI client for interacting with the API.
        messages (list): A list to keep track of the conversation history.
    """

    def __init__(self) -> None:
        """
        Initializes a new Completions instance, loading environment variables
        and creating an OpenAI client for API interactions.
        """
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        self.db = self.mongo_client.chat_db
        self.conversations_collection = self.db.conversations

        load_dotenv()

        with open(".instructions.json", "r", encoding="utf-8") as json_file:
            instructions = json.load(json_file)["instructions"]

        self.messages = [{"role": "system", "content": instructions}]

        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
        )

    def get_message(self, message: str, isJson: bool = False) -> str:
        """
        Sends a message to the OpenAI API and stores the response.

        Parameters:
            message (str): The user's input message to send to the API.

        Returns:
            str: The response message from the OpenAI completion model.

        Raises:
            TypeError: If the response from the API is not a string.
        """
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        if isJson:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),  # type: ignore
                messages=self.messages,  # type: ignore
                response_format={"type": "json_object"},
            )
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),  # type: ignore
                messages=self.messages,  # type: ignore
            )

        got_message = response.choices[0].message.content

        if got_message is None:
            raise TypeError("Received message is not a string")

        self.messages.append(
            {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
        )

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
        Saves the conversation summary to MongoDB.

        Parameters:
            conversation_summary (dict): The summary of the conversation.
        """
        self.conversations_collection.insert_one(conversation_summary)
        print("Conversation saved to MongoDB.")

    def chat(self) -> None:
        """
        Initiates a continuous chat interaction with the OpenAI chat model.

        In this conversational loop, the method prompts the user for input, sends
        it to the OpenAI API, receives a response, and displays it. The loop
        continues until the user types 'exit', at which point the conversation
        terminates.
        """

        while True:
            message = self.get_user_input()

            # To exit the conversation
            if message == "exit":
                break

            # To clear the conversation history
            if message == "clear":
                self.messages = []
                continue

            # To save the conversation summary to MongoDB
            if message == "save":
                summary_instruction = """
                Summarize the conversation so far in the following json format.


                {"title":str,"summary":str}
                """

                summary = self.get_message(summary_instruction, True)
                print(f"Summary : \n\n{summary}\n")

                summary_json = json.loads(summary)
                self.save_conversation_to_mongo(summary_json)
                break

            # To get the next message from the assistant
            response = self.get_message(message)
            print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    Completions().chat()
