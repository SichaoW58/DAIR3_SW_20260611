"""
agentGoogle.py
Command-line chatbot interface using the Google Gemini API.
Configuration is dynamically loaded from a JSON file with user and assistant properties.
Includes support for file uploads (via the Files API) and persistent chat sessions.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""

import os
import json
import sys
import io
import time

from google import genai
from google.genai import types

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class GoogleChatbot:
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

        # Allow CONFIG to override which Gemini model to use for this CLI
        self.model = config.get('google_model', 'gemini-2.5-flash')

        # Get the Gemini API key from the environment and validate it
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("API key is not set. Please set GEMINI_API_KEY (or GOOGLE_API_KEY).")
            exit(1)

        # Initialize the Gemini client and a stateful chat session.
        # The chat session retains turn history internally across send_message calls.
        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=self.instructions),
        )

    def upload_file(self, file_path):
        """
        Upload a local file via the Gemini Files API and wait until it is ACTIVE
        so it can be attached to a subsequent message. Returns the file object.
        """
        try:
            my_file = self.client.files.upload(file=file_path)
            # Wait while the file is still being processed
            while getattr(my_file, "state", None) and str(my_file.state).endswith("PROCESSING"):
                time.sleep(1)
                my_file = self.client.files.get(name=my_file.name)
            print(f"File uploaded successfully: {my_file.name}")
            return my_file
        except Exception as e:
            print(f"Failed to upload file: {e}")
            return None

    def run_chat(self):
        # Display introductory details about the session
        print("*****************   N E W   C H A T   *****************")
        print(f"Model: {self.model}")
        print(f"Agent: {self.name}")

        # Main chat loop
        while True:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
            user_input = input(f"{self.user}: ")

            # Exit condition
            if user_input.lower() == 'exit':
                break

            # Handle file upload request: upload + acknowledgement turn
            if user_input.startswith("file:"):
                file_path = user_input[5:].strip()
                my_file = self.upload_file(file_path)
                if my_file:
                    try:
                        print("Thinking...", end="", flush=True)
                        response = self.chat.send_message([
                            my_file,
                            "Please acknowledge this file. I will ask follow-up questions about it.",
                        ])
                        print()
                        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        print("\n" + self.name + f": {response.text}")
                    except Exception as e:
                        print(f"Failed to send file to model: {e}")
                continue

            try:
                print("Thinking...", end="", flush=True)
                response = self.chat.send_message(user_input)
                print()
                print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("\n" + self.name + f": {response.text}")
            except Exception as e:
                print(f"Error: {e}")


# Entry point of the program
if __name__ == "__main__":
    agent = GoogleChatbot()
    agent.run_chat()
