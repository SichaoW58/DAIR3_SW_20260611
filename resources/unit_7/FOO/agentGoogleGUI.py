"""
agentGoogleGUI.py
GUI chatbot interface using the Google Gemini API.
Configuration is dynamically loaded from a JSON file with user and assistant properties.
Includes support for file uploads (drag-and-drop via the Files API) and a persistent chat session.

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
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QLineEdit, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QClipboard

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class LLMWorker(QThread):
    # Define a signal to emit when the model's response is ready
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, chat):
        super().__init__()
        self.user_input = user_input
        self.chat = chat  # stateful chat session owned by the parent widget

    def run(self):
        try:
            response = self.chat.send_message(self.user_input)
            self.result_ready.emit(response.text)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")


class GoogleChatbot(QWidget):
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
        self.model = config.get('google_model', 'gemini-2.5-flash')
        self.latest_response = ""

        # Load API key from environment
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("API key is not set. Please set GEMINI_API_KEY (or GOOGLE_API_KEY).")
            exit(1)

        # Initialize Gemini client and a stateful chat session
        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=self.instructions),
        )

        # Set up the GUI interface
        self.init_gui()

    def init_gui(self):
        # Set up main window parameters
        self.setWindowTitle("JuanGemini")
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
        # Upload a file via Gemini Files API and consume it with an acknowledgement turn
        try:
            my_file = self.client.files.upload(file=file_path)
            while getattr(my_file, "state", None) and str(my_file.state).endswith("PROCESSING"):
                time.sleep(1)
                my_file = self.client.files.get(name=my_file.name)
            self.text_area.append(f"File uploaded: {my_file.name}")

            try:
                response = self.chat.send_message([
                    my_file,
                    "Please acknowledge this file. I will ask follow-up questions about it.",
                ])
                self.text_area.append(f"{self.name}: {response.text}")
                self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")
            except Exception as e:
                self.text_area.append(f"Failed to send file to model: {e}")
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
        self.worker_thread = LLMWorker(user_input, self.chat)
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
    chatbot = GoogleChatbot()
    chatbot.show()
    app.exec_()
