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
    QPushButton, QTabWidget, QHBoxLayout, QCheckBox, QLabel, QScrollArea,
    QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal

# Force UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

HARMONIZER_FALLBACK_DIRECTIVE = (
    "The following statements are the flaws others found for agent {source_agent_name}'s response."
    " Organize their responses by topic in an additive manner (that is, do not eliminate information)."
    " Structure your response using the following sections: 'Agreement', 'Disagreement', and 'Unique observations'."
    " In 'Agreement', list ideas supported by multiple agents. In 'Disagreement', note contradictory statements."
    " In 'Unique observations', highlight observations made by only one agent."
    " The agent under review needs detailed responses to be able to improve."
    " Produce the content for these sections with detailed bulletpoints."
)


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
    log_signal = pyqtSignal(str)

    def __init__(self, model, name, instructions, user, engine, harmonizer=False, harmonizer_directive=""):
        super().__init__()
        self.user = user
        self.name = name
        self.model = model
        self.engine = engine
        self.harmonizer = harmonizer
        self.harmonizer_directive = harmonizer_directive or ""
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

        role_text = "Harmonizer" if self.harmonizer else "Reviewer"
        self.checkbox = QCheckBox(f"Enable {self.name} ({role_text})")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.toggle_active)
        layout.addWidget(self.checkbox)

        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        self.user_input.setPlaceholderText("Type your message and press Enter")
        layout.addWidget(self.user_input)

        self.copy_button.clicked.connect(self.copy_latest_answer)
        layout.addWidget(self.copy_button)

        self.log_signal.connect(self.text_area.append)

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

    def send_message_sync(self, text, sender_label=None):
        sender = sender_label or self.user
        self.log_signal.emit(f"{sender}: {text}")
        self.log_signal.emit(">>>>>>>>>>>>>>>>>>>>>>>>>>")
        try:
            if self.engine == "openai":
                response = self._openai_sync(text)
            elif self.engine == "google":
                response = self._google_sync(text)
            else:
                response = self._claude_sync(text)
        except Exception as e:
            response = f"Error: {e}"
        self.latest_response = response
        self.log_signal.emit(f"{self.name}: {response}")
        self.log_signal.emit("<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return response

    def _openai_sync(self, text):
        kwargs = {
            "model": self.model,
            "instructions": self.instructions,
            "input": text,
        }
        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id
        response = self.client.responses.create(**kwargs)
        self.previous_response_id = response.id
        return response.output_text

    def _google_sync(self, text):
        return self.chat.send_message(text).text

    def _claude_sync(self, text):
        self.history.append({"role": "user", "content": text})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.99,
            messages=self.history,
        )
        content = response.content[0].text
        self.history.append({"role": "assistant", "content": content})
        return content


class FOOOrchestratorWorker(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, user_text, non_harmonizers, harmonizers, iterations):
        super().__init__()
        self.user_text = user_text
        self.non_harmonizers = non_harmonizers
        self.harmonizers = harmonizers
        self.iterations = iterations

    def run(self):
        for tab in self.non_harmonizers:
            tab.log_signal.emit("--- Initial Broadcast ---")
            tab.send_message_sync(self.user_text)

        if self.iterations <= 0:
            self.finished_signal.emit()
            return

        if len(self.non_harmonizers) < 2:
            for tab in self.non_harmonizers:
                tab.log_signal.emit(
                    "[FOO] Need at least 2 active non-harmonizer agents for critique cycles. Stopping after broadcast."
                )
            self.finished_signal.emit()
            return

        for cycle in range(1, self.iterations + 1):
            for source in self.non_harmonizers:
                vuln_msg = (
                    f"Agent {source.name} answered the same question as follows, find flaws: "
                    f"{source.latest_response}"
                )
                critiques = {}
                for reviewer in self.non_harmonizers:
                    if reviewer is source:
                        continue
                    reviewer.log_signal.emit(
                        f"--- Cycle {cycle} | Vulnerability: critique of {source.name} ---"
                    )
                    critiques[reviewer.name] = reviewer.send_message_sync(vuln_msg)

                composite = "".join(
                    f"\n \n Agent {name}: {resp}" for name, resp in critiques.items()
                )

                harmonizer_outputs = []
                for harmonizer in self.harmonizers:
                    if harmonizer.harmonizer_directive:
                        directive = harmonizer.harmonizer_directive.replace(
                            "{source_agent_name}", source.name
                        )
                    else:
                        directive = HARMONIZER_FALLBACK_DIRECTIVE.format(
                            source_agent_name=source.name
                        )
                    judgment_msg = f"{directive} \n \n {composite}"
                    harmonizer.log_signal.emit(
                        f"--- Cycle {cycle} | Judgment on {source.name} ---"
                    )
                    harmonizer_outputs.append(harmonizer.send_message_sync(judgment_msg))

                reflection_input = (
                    "---".join(harmonizer_outputs) if harmonizer_outputs else composite
                )
                reflection_msg = (
                    "Judgment of your response has resulted in the observations that follow. "
                    "Regenerate your version of the text under review taking into account the consensus of these observations. "
                    "If you object to an observation, explain why. \n \n " + reflection_input
                )
                source.log_signal.emit(
                    f"--- Cycle {cycle} | Reflection on {source.name} ---"
                )
                source.send_message_sync(reflection_msg)

        self.finished_signal.emit()


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

            harmonizer_raw = entry.get("harmonizer", False)
            if isinstance(harmonizer_raw, bool):
                harmonizer = harmonizer_raw
            else:
                harmonizer = str(harmonizer_raw).strip().lower() == "true"

            tab = AgentTab(
                model=model_code,
                name=entry["agent_name"],
                instructions=config["instructions"],
                user=self.user,
                engine=engine,
                harmonizer=harmonizer,
                harmonizer_directive=entry.get("harmonizer_directive", "") or "",
            )
            self.tabs.addTab(tab, entry["agent_name"])
            self.agent_tabs.append(tab)

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Broadcast message to all active agents")
        self.user_input.returnPressed.connect(self.broadcast_message)

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(0, 20)
        self.iterations_spin.setValue(3)
        self.iterations_spin.setFixedWidth(60)

        input_row = QHBoxLayout()
        input_row.addWidget(self.user_input, 1)
        input_row.addWidget(QLabel("FOO Cycles:"))
        input_row.addWidget(self.iterations_spin)

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Message to All Active Agents:"))
        layout.addLayout(input_row)
        self.setLayout(layout)

        self.setWindowTitle("Multi-Agent GPT + Claude + Gemini Interface")
        self.resize(700, 500)

        self.orchestrator = None

    def broadcast_message(self):
        text = self.user_input.text().strip()
        if not text:
            return
        if self.orchestrator is not None and self.orchestrator.isRunning():
            return

        active = [t for t in self.agent_tabs if t.active]
        non_harmonizers = [t for t in active if not t.harmonizer]
        harmonizers = [t for t in active if t.harmonizer]

        if not non_harmonizers:
            return

        self.user_input.clear()
        self.user_input.setEnabled(False)
        self.user_input.setPlaceholderText("FOO run in progress...")

        self.orchestrator = FOOOrchestratorWorker(
            user_text=text,
            non_harmonizers=non_harmonizers,
            harmonizers=harmonizers,
            iterations=self.iterations_spin.value(),
        )
        self.orchestrator.finished_signal.connect(self.on_foo_finished)
        self.orchestrator.start()

    def on_foo_finished(self):
        self.user_input.setEnabled(True)
        self.user_input.setPlaceholderText("Broadcast message to all active agents")

if __name__ == "__main__":
    app = QApplication([])
    window = MultiAgentChat()
    window.show()
    app.exec_()
