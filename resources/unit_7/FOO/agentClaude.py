import os
import sys
import json
import io
import openai
import anthropic
from google import genai
from google.genai import types
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QLineEdit, QVBoxLayout,
    QPushButton, QTabWidget, QHBoxLayout, QCheckBox, QLabel, QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class OpenAIWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, client, model, instructions, tab):
        super().__init__()
        self.user_input = user_input
        self.client = client
        self.model = model
        self.instructions = instructions
        self.tab = tab  # owns previous_response_id for conversation continuity

    def run(self):
        try:
            kwargs = {
                "model": self.model,
                "instructions": self.instructions,
                "input": self.user_input,
            }
            if self.tab.previous_response_id:
                kwargs["previous_response_id"] = self.tab.previous_response_id

            response = self.client.responses.create(**kwargs)
            self.tab.previous_response_id = response.id
            self.result_ready.emit(response.output_text)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")

class GoogleWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, chat):
        super().__init__()
        self.user_input = user_input
        self.chat = chat  # stateful chat session owned by the tab

    def run(self):
        try:
            response = self.chat.send_message(self.user_input)
            self.result_ready.emit(response.text)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")

class ClaudeWorker(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, client, model, history):
        super().__init__()
        self.user_input = user_input
        self.client = client
        self.model = model
        self.history = history

    def run(self):
        try:
            self.history.append({"role": "user", "content": self.user_input})
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.99,
                messages=self.history
            )
            content = response.content[0].text
            self.history.append({"role": "assistant", "content": content})
            self.result_ready.emit(content)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")

class AgentTab(QWidget):
    def __init__(self, model, name, instructions, user, engine):
        super().__init__()
        self.user = user
        self.name = name
        self.model = model
        self.engine = engine
        self.latest_response = ""
        self.active = True  # Controlled by checkbox

        self.text_area = QTextEdit()
        self.user_input = QLineEdit()
        self.copy_button = QPushButton("Copy Latest Answer")

        if self.engine == "openai":
            preamble = f"Please address the user as Beloved {user}.\\n\\n Introduce yourself as {name}, robot extraordinaire.\\n\\n "
            self.instructions = preamble + instructions
            self.client = openai.OpenAI()
            self.previous_response_id = None
        elif self.engine == "google":
            preamble = f"Please address the user as Beloved {user}.\\n\\n Introduce yourself as {name}, robot extraordinaire.\\n\\n "
            self.instructions = preamble + instructions
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
            self.chat = self.client.chats.create(
                model=model,
                config=types.GenerateContentConfig(system_instruction=self.instructions),
            )
        else:
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.history = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.checkbox = QCheckBox(f"Enable {self.name}")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_active)
        layout.addWidget(self.checkbox)

        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        self.user_input.setPlaceholderText("Type your message and press Enter")
        layout.addWidget(self.user_input)

        self.copy_button.clicked.connect(self.copy_latest_answer)
        layout.addWidget(self.copy_button)

        self.setLayout(layout)

    def toggle_active(self, state):
        self.active = bool(state)

    def handle_input(self, text):
        if not self.active:
            return

        self.text_area.append(f"{self.user}: {text}")
        self.text_area.append(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.user_input.setEnabled(False)

        if self.engine == "openai":
            self.worker = OpenAIWorker(text, self.client, self.model, self.instructions, self)
        elif self.engine == "google":
            self.worker = GoogleWorker(text, self.chat)
        else:
            self.worker = ClaudeWorker(text, self.client, self.model, self.history)

        self.worker.result_ready.connect(self.show_response)
        self.worker.start()

    def show_response(self, response):
        self.latest_response = response
        self.text_area.append(f"{self.name}: {response}")
        self.text_area.append("<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.user_input.setEnabled(True)

    def copy_latest_answer(self):
        QApplication.clipboard().setText(self.latest_response)
        self.text_area.append("Latest answer copied to clipboard.")

class MultiAgentChat(QWidget):
    def __init__(self):
        super().__init__()
        with open("config.json", "r") as f:
            config_data = json.load(f)
        config = config_data["CONFIG"]
        models = config_data["MODELS"]

        self.user = config["user"]

        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            print("API key is not set. Please set the OPENAI_API_KEY environment variable.")
            exit(1)

        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Warning: ANTHROPIC_API_KEY not set. Claude agents will not function.")

        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            print("Warning: GEMINI_API_KEY/GOOGLE_API_KEY not set. Gemini agents will not function.")

        self.tabs = QTabWidget()
        self.agent_tabs = []

        for entry in models:
            model_code = entry["model_code"]
            if model_code.startswith("claude"):
                engine = "claude"
            elif model_code.startswith("gemini"):
                engine = "google"
            else:
                engine = "openai"
            tab = AgentTab(
                model=model_code,
                name=entry["agent_name"],
                instructions=config["instructions"],
                user=self.user,
                engine=engine
            )
            self.tabs.addTab(tab, entry["agent_name"])
            self.agent_tabs.append(tab)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Broadcast message to all active agents")
        self.user_input.returnPressed.connect(self.broadcast_message)

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Message to All Active Agents:"))
        layout.addWidget(self.user_input)
        self.setLayout(layout)

        self.setWindowTitle("Multi-Agent GPT + Claude + Gemini Interface")
        self.resize(700, 500)

    def broadcast_message(self):
        text = self.user_input.text().strip()
        if not text:
            return
        for tab in self.agent_tabs:
            tab.handle_input(text)
        self.user_input.clear()

if __name__ == "__main__":
    app = QApplication([])
    window = MultiAgentChat()
    window.show()
    app.exec_()
