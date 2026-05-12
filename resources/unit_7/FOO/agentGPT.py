"""
Helper.py
Command-line chatbot interface using the OpenAI Responses API.
Configuration is dynamically loaded from a JSON file with user and assistant properties.
Includes support for file uploads (via vector stores + file_search) and persistent
conversation state (via previous_response_id).

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""

import os
import openai
import json
import sys
import io

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class OpenAIChatbot:
    def __init__(self, config_file="config.json"):
        # Load the JSON configuration from the provided file
        with open(config_file, 'r') as file:
            raw_config = json.load(file)
            config = raw_config['CONFIG']  # Focus on the CONFIG section only

        # Extract user and assistant name from configuration
        self.user = config['user']
        self.name = config['name']

        # Prepend custom interaction preamble to instruction text
        preamble = f"Please address the user as Beloved {self.user}.\\n\\n Introduce yourself as {self.name}, robot extraordinaire.\\n\\n "
        self.instructions = preamble + config['instructions']

        # Extract the model identifier used for the assistant
        self.model = config['model']

        # Get the OpenAI API key from the environment and validate it
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            print("API key is not set. Please set the OPENAI_API_KEY environment variable.")
            exit(1)

        # Initialize the OpenAI API client
        self.client = openai.OpenAI()

        # Vector store backing the file_search tool (created lazily on first upload)
        self.vector_store_id = None

        # Tracks Responses API conversation state across turns
        self.previous_response_id = None

    def upload_file(self, file_path):
        """
        Uploads a local file to the OpenAI API and registers it with a vector store
        so the file_search tool can retrieve from it. Returns the file ID on success.
        """
        try:
            with open(file_path, 'rb') as file_data:
                file_object = self.client.files.create(
                    file=file_data,
                    purpose='assistants'
                )
            print(f"File uploaded successfully: ID {file_object.id}")

            # Create the vector store on the first upload
            if self.vector_store_id is None:
                vs = self.client.vector_stores.create(name=f"{self.name}_files")
                self.vector_store_id = vs.id

            # Attach file to the vector store and wait until it is indexed
            self.client.vector_stores.files.create_and_poll(
                vector_store_id=self.vector_store_id,
                file_id=file_object.id,
            )
            return file_object.id
        except Exception as e:
            print(f"Failed to upload file: {e}")
            return None

    def _tools(self):
        if self.vector_store_id:
            return [{"type": "file_search", "vector_store_ids": [self.vector_store_id]}]
        return None

    def run_chat(self):
        # Display introductory details about the session
        print("*****************   N E W   C H A T   *****************")
        print(f"Model: {self.model}")
        print(f"Agent: {self.name}")

        # Main chat loop
        while True:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
            # Prompt user input
            user_input = input(f"{self.user}: ")

            # Exit condition
            if user_input.lower() == 'exit':
                break

            # Handle file upload request
            if user_input.startswith("file:"):
                file_path = user_input[5:].strip()
                file_id = self.upload_file(file_path)
                if file_id:
                    print(f"File ID {file_id} indexed in vector store {self.vector_store_id}")
                continue

            try:
                # Build request, only including optional fields when set
                kwargs = {
                    "model": self.model,
                    "instructions": self.instructions,
                    "input": user_input,
                }
                if self.previous_response_id:
                    kwargs["previous_response_id"] = self.previous_response_id
                tools = self._tools()
                if tools:
                    kwargs["tools"] = tools

                print("Thinking...", end="", flush=True)
                response = self.client.responses.create(**kwargs)
                print()

                # Remember response id so the next turn keeps the conversation thread
                self.previous_response_id = response.id

                print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("\n" + self.name + f": {response.output_text}")
            except Exception as e:
                print(f"Error: {e}")

# Entry point of the program
if __name__ == "__main__":
    agent = OpenAIChatbot()
    agent.run_chat()
