"""
HelperGUI.py
GUI chatbot interface using the OpenAI API and assistant capabilities.
Configuration is dynamically loaded from a JSON file with user and assistant properties.
Includes support for file uploads and persistent threaded interactions.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import os
import openai
import json
import sys
import io
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QLineEdit, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QClipboard

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class LLMWorker(QThread):
    # Define a signal to emit when the assistant's response is ready
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, chatbot):
        super().__init__()
        self.user_input = user_input
        self.chatbot = chatbot  # owns client, model, instructions, conversation state

    def run(self):
        try:
            kwargs = {
                "model": self.chatbot.model,
                "instructions": self.chatbot.instructions,
                "input": self.user_input,
            }
            if self.chatbot.previous_response_id:
                kwargs["previous_response_id"] = self.chatbot.previous_response_id
            if self.chatbot.vector_store_id:
                kwargs["tools"] = [{
                    "type": "file_search",
                    "vector_store_ids": [self.chatbot.vector_store_id],
                }]

            response = self.chatbot.client.responses.create(**kwargs)
            self.chatbot.previous_response_id = response.id
            self.result_ready.emit(response.output_text)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")

class OpenAIChatbot(QWidget):
    def __init__(self):
        super().__init__()

        # Load configuration from CONFIG section
        config_file = "config.json"
        with open(config_file, 'r') as file:
            raw_config = json.load(file)
            config = raw_config['CONFIG']

        # Extract configuration values
        self.user = config['user']
        self.name = config['name']

        # Build instructions with dynamic user and assistant introduction
        preamble = f"Please address the user as Beloved {self.user}.\\n\\n Introduce yourself as {self.name}, robot extraordinaire.\\n\\n "
        self.instructions = preamble + config['instructions']
        self.model = config['model']
        self.latest_response = ""

        # Load API key from environment
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            print("API key is not set. Please set the OPENAI_API_KEY environment variable.")
            exit(1)

        # Create OpenAI API client
        self.client = openai.OpenAI()

        # Responses API conversation pointer; updated after every successful turn.
        self.previous_response_id = None

        # Vector store backing the file_search tool; created lazily on first upload.
        self.vector_store_id = None

        # Set up the GUI interface
        self.init_gui()

    def init_gui(self):
        # Set up main window parameters
        self.setWindowTitle("JuanGPT")
        self.setGeometry(100, 100, 600, 400)
        self.setAcceptDrops(True)  # Enable drag and drop

        layout = QVBoxLayout()

        # Text display area for messages
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        # Display session header
        self.text_area.append(f"Model: {self.model}")
        self.text_area.append(f"Agent: {self.name}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")

        # User input field
        self.user_input = QLineEdit(self)
        self.user_input.setPlaceholderText("Type your message and press Enter")
        layout.addWidget(self.user_input)

        # Button to copy latest assistant response
        self.copy_button = QPushButton("Copy Latest Answer")
        self.copy_button.clicked.connect(self.copy_latest_answer)
        layout.addWidget(self.copy_button)

        # Bind Enter key to user input processing
        self.user_input.returnPressed.connect(self.on_enter_pressed)

        # Apply layout to the window
        self.setLayout(layout)

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept drag event if a file is present
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        # Handle file drop by extracting path
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.upload_file(file_path)

    def upload_file(self, file_path):
        # Upload a file and register it with the file_search vector store
        try:
            with open(file_path, 'rb') as file_data:
                file_object = self.client.files.create(
                    file=file_data,
                    purpose='assistants'
                )
            self.text_area.append(f"File uploaded successfully: ID {file_object.id}")

            try:
                if self.vector_store_id is None:
                    vs = self.client.vector_stores.create(name=f"{self.name}_files")
                    self.vector_store_id = vs.id

                self.client.vector_stores.files.create_and_poll(
                    vector_store_id=self.vector_store_id,
                    file_id=file_object.id,
                )
                self.text_area.append(f"File indexed in vector store {self.vector_store_id}")
            except Exception as e:
                self.text_area.append(f"Failed to index file for search: {e}")

            self.text_area.append(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        except Exception as e:
            self.text_area.append(f"Failed to upload file: {e}")

    def on_enter_pressed(self):
        # Process input when Enter is pressed
        user_input = self.user_input.text().strip()
        if user_input:
            self.process_user_input(user_input)
        self.user_input.clear()

    def process_user_input(self, user_input):
        # Display user input
        self.text_area.append(f"{self.user}: {user_input}")
        self.text_area.append(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.user_input.setEnabled(False)

        # Launch worker thread to handle assistant response
        self.worker_thread = LLMWorker(user_input, self)
        self.worker_thread.result_ready.connect(self.display_results)
        self.worker_thread.start()

    def display_results(self, response):
        # Show assistant response
        self.latest_response = response
        self.text_area.append(f"{self.name}: {response}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.user_input.setEnabled(True)

    def copy_latest_answer(self):
        # Copy latest assistant message to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(self.latest_response)
        self.text_area.append("Latest answer copied to clipboard.")

# Start the GUI application
if __name__ == "__main__":
    app = QApplication([])
    chatbot = OpenAIChatbot()
    chatbot.show()
    app.exec_()
