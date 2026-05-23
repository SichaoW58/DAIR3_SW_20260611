"""
grant_review_v2.py
Two-agent interface for collaborative analysis with improved layout and controls.
Supports vulnerability analysis and reflection between agents.

Based on the "Flaws of Others" methodology for grant review consensus.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

VERSION 2: Updated layout with proportional sizing and font controls
"""

import os
import sys
import json
import base64
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox, QSplitter,
    QGroupBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtMultimedia import QSound

from cls_foo import MultiAgentOrchestrator
from cls_openai import OpenAIAgent
from cls_anthropic import AnthropicAgent


class BroadcastTextEdit(QTextEdit):
    """Custom QTextEdit for broadcast messages"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setPlaceholderText("Type your question/message to broadcast to both agents (Shift+Enter for new line, Enter to send)")
        
    def keyPressEvent(self, event):
        """Handle Enter key for broadcasting"""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                text = self.toPlainText().strip()
                if text and self.parent_widget:
                    self.parent_widget.broadcast_to_agents(text)
                    self.clear()
                return
        super().keyPressEvent(event)


class AgentTextEdit(QTextEdit):
    """Custom QTextEdit for individual agent input"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.agent_panel = parent
        self.setPlaceholderText("Type message for this agent (Shift+Enter for new line, Enter to send)")
        
    def keyPressEvent(self, event):
        """Handle Enter key for individual messages"""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                text = self.toPlainText().strip()
                if text and self.agent_panel:
                    self.agent_panel.send_individual_message(text)
                    self.clear()
                return
        super().keyPressEvent(event)


class AgentWorker(QThread):
    """Worker thread for agent message processing using orchestrator"""
    result_ready = pyqtSignal(str)
    
    def __init__(self, orchestrator, agent_name, message):
        super().__init__()
        self.orchestrator = orchestrator
        self.agent_name = agent_name
        self.message = message
    
    def run(self):
        try:
            # Get the agent directly and use send_message_with_integrity
            agent = self.orchestrator.get_agent_by_name(self.agent_name)
            if agent:
                response = self.orchestrator.send_message_with_integrity(agent, self.message)
                self.result_ready.emit(response)
            else:
                self.result_ready.emit(f"Error: Agent {self.agent_name} not found")
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")


class BroadcastWorker(QThread):
    """Worker thread for broadcast messages using orchestrator"""
    results_ready = pyqtSignal(dict)
    
    def __init__(self, orchestrator, message):
        super().__init__()
        self.orchestrator = orchestrator
        self.message = message
    
    def run(self):
        try:
            # Use orchestrator's broadcast_message which handles history
            responses = self.orchestrator.broadcast_message(self.message)
            self.results_ready.emit(responses)
        except Exception as e:
            self.results_ready.emit({"Error": str(e)})


class VulnerabilityWorker(QThread):
    """Worker thread for vulnerability analysis using orchestrator"""
    result_ready = pyqtSignal(str, str)  # (request, response)
    
    def __init__(self, orchestrator, source_agent_name, target_agent_name):
        super().__init__()
        self.orchestrator = orchestrator
        self.source_agent_name = source_agent_name
        self.target_agent_name = target_agent_name
    
    def run(self):
        try:
            source_agent = self.orchestrator.get_agent_by_name(self.source_agent_name)
            if not source_agent or not source_agent.latest_response:
                self.result_ready.emit("", "No response available from the other agent to analyze")
                return
            
            # Create vulnerability analysis prompt
            request_message = (
                f"The other agent ({self.source_agent_name}) provided the following response. "
                f"Your task is to critically analyze this response and identify any flaws, "
                f"weaknesses, unsupported claims, logical inconsistencies, or areas that need improvement:\n\n"
                f"{source_agent.latest_response}"
            )
            
            # Use send_message_with_integrity with the agent object
            target_agent = self.orchestrator.get_agent_by_name(self.target_agent_name)
            if target_agent:
                response = self.orchestrator.send_message_with_integrity(target_agent, request_message)
                self.result_ready.emit(request_message, response)
            else:
                self.result_ready.emit("", f"Error: Agent {self.target_agent_name} not found")
            
        except Exception as e:
            self.result_ready.emit("", f"Error: {e}")


class ReflectionWorker(QThread):
    """Worker thread for reflection analysis using orchestrator"""
    result_ready = pyqtSignal(str, str)  # (request, response)
    
    def __init__(self, orchestrator, source_agent_name, target_agent_name):
        super().__init__()
        self.orchestrator = orchestrator
        self.source_agent_name = source_agent_name  # Agent providing critique
        self.target_agent_name = target_agent_name  # Agent reflecting on its own work
    
    def run(self):
        try:
            source_agent = self.orchestrator.get_agent_by_name(self.source_agent_name)
            if not source_agent or not source_agent.latest_response:
                self.result_ready.emit("", "No critique available from the other agent")
                return
            
            # Create reflection prompt
            request_message = (
                f"The other agent ({self.source_agent_name}) has provided the following observations "
                f"and critique of your previous response. Please reflect on this feedback and regenerate "
                f"your response, addressing the valid concerns while explaining why you might disagree "
                f"with any points you find incorrect:\n\n"
                f"{source_agent.latest_response}"
            )
            
            # Use send_message_with_integrity with the agent object
            target_agent = self.orchestrator.get_agent_by_name(self.target_agent_name)
            if target_agent:
                response = self.orchestrator.send_message_with_integrity(target_agent, request_message)
                self.result_ready.emit(request_message, response)
            else:
                self.result_ready.emit("", f"Error: Agent {self.target_agent_name} not found")
            
        except Exception as e:
            self.result_ready.emit("", f"Error: {e}")


class FileProcessorWorker(QThread):
    """Worker thread for processing various file types"""
    result_ready = pyqtSignal(str, bool)  # (content/message, should_broadcast)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            filename = os.path.basename(self.file_path)
            
            if ext == '.pdf':
                # Try to extract text from PDF
                try:
                    import PyPDF2
                    with open(self.file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        total_pages = len(pdf_reader.pages)
                        pdf_text = f"[PDF: {filename}, {total_pages} pages]\n\n"
                        
                        text_found = False
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            text = page.extract_text()
                            if text.strip():
                                text_found = True
                                pdf_text += f"\n--- Page {page_num} ---\n{text}\n"
                        
                        if not text_found:
                            self.result_ready.emit(
                                f"❌ No text could be extracted from PDF: {filename}\n"
                                "This might be a scanned document or contain only images.",
                                False
                            )
                        else:
                            self.result_ready.emit(pdf_text, True)
                            
                except ImportError:
                    self.result_ready.emit(
                        "Error: PyPDF2 not installed. Please install it with:\n"
                        "pip install PyPDF2",
                        False
                    )
                except Exception as e:
                    self.result_ready.emit(f"Error processing PDF: {str(e)}", False)
                    
            elif ext == '.txt':
                # Load text file
                try:
                    with open(self.file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    self.result_ready.emit(f"[Text file: {filename}]\n\n{content}", True)
                except Exception as e:
                    self.result_ready.emit(f"Error reading text file: {str(e)}", False)
                    
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # Handle image files
                try:
                    # Convert to base64 for potential OCR or image analysis
                    with open(self.file_path, 'rb') as file:
                        img_data = base64.b64encode(file.read()).decode()
                    
                    # For now, just inform that image was loaded
                    self.result_ready.emit(
                        f"[Image file: {filename}]\n"
                        f"Image loaded successfully. Size: {os.path.getsize(self.file_path) / 1024:.1f} KB\n"
                        f"Note: Image analysis capabilities depend on agent configuration.",
                        True
                    )
                except Exception as e:
                    self.result_ready.emit(f"Error processing image: {str(e)}", False)
                    
            elif ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                # Handle sound files
                try:
                    self.result_ready.emit(
                        f"[Audio file: {filename}]\n"
                        f"Audio file detected. Size: {os.path.getsize(self.file_path) / 1024:.1f} KB\n"
                        f"Note: Audio transcription capabilities depend on agent configuration.",
                        True
                    )
                except Exception as e:
                    self.result_ready.emit(f"Error processing audio: {str(e)}", False)
                    
            else:
                self.result_ready.emit(
                    f"Unsupported file type: {ext}\n"
                    f"Supported types: PDF, TXT, PNG, JPG, JPEG, GIF, BMP, MP3, WAV, OGG, M4A",
                    False
                )
                
        except Exception as e:
            self.result_ready.emit(f"Error processing file: {str(e)}", False)


class AgentPanel(QGroupBox):
    """Panel for individual agent interaction with vulnerability/reflection controls"""
    
    def __init__(self, agent, orchestrator, panel_number, parent_gui):
        super().__init__()
        self.agent = agent
        self.orchestrator = orchestrator
        self.panel_number = panel_number
        self.parent_gui = parent_gui
        self.other_panel = None
        
        # Worker references
        self.worker = None
        self.vulnerability_worker = None
        self.reflection_worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI for the agent panel"""
        layout = QVBoxLayout()
        
        # Agent selector
        selector_layout = QHBoxLayout()
        self.agent_label = QLabel("Agent:")
        selector_layout.addWidget(self.agent_label)
        self.agent_selector = QComboBox()
        self.populate_agent_selector()
        self.agent_selector.currentIndexChanged.connect(self.on_agent_changed)
        selector_layout.addWidget(self.agent_selector, 1)
        layout.addLayout(selector_layout)
        
        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area, 3)
        
        # Individual message input
        self.input_area = AgentTextEdit(self)
        layout.addWidget(self.input_area, 1)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.vulnerability_btn = QPushButton("🔍 Vulnerability")
        self.vulnerability_btn.setToolTip("Have the other agent critique this agent's response")
        self.vulnerability_btn.clicked.connect(self.on_vulnerability_clicked)
        button_layout.addWidget(self.vulnerability_btn)
        
        self.reflection_btn = QPushButton("🔄 Reflection")
        self.reflection_btn.setToolTip("Have this agent reflect on the other's critique and improve response")
        self.reflection_btn.clicked.connect(self.on_reflection_clicked)
        button_layout.addWidget(self.reflection_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.update_title()
        self.update_fonts()
    
    def populate_agent_selector(self):
        """Populate the agent selector dropdown"""
        self.agent_selector.clear()
        
        # Get all active agents
        agents = [a for a in self.orchestrator.agents if a.active]
        
        for agent in agents:
            self.agent_selector.addItem(agent.name, agent)
        
        # Set current agent if provided
        if self.agent:
            for i in range(self.agent_selector.count()):
                if self.agent_selector.itemData(i) == self.agent:
                    self.agent_selector.setCurrentIndex(i)
                    break
    
    def on_agent_changed(self, index):
        """Handle agent selection change - load that agent's log"""
        self.agent = self.agent_selector.itemData(index)
        self.update_title()
        self.output_area.clear()
        
        if self.agent:
            # Load the agent's conversation history
            self.load_agent_history()
    
    def load_agent_history(self):
        """Load and display the selected agent's conversation history"""
        if not self.agent:
            return
            
        try:
            # Check if agent has history
            if hasattr(self.agent, 'display_history') and self.agent.display_history:
                self.output_area.append("=== LOADING CONVERSATION HISTORY ===\n")
                
                for entry in self.agent.display_history:
                    if isinstance(entry, dict):
                        role = entry.get('role', 'unknown')
                        content = entry.get('content', '')
                        timestamp = entry.get('timestamp', '')
                        
                        # Skip initial system messages
                        if role == 'user' and 'Introduce yourself as' in content:
                            continue
                        
                        time_str = f" ({timestamp})" if timestamp else ""
                        
                        if role == 'user':
                            self.output_area.append(f"\n📤 User{time_str}:")
                            self.output_area.append(content)
                        elif role == 'assistant':
                            self.output_area.append(f"\n💬 {self.agent.name}{time_str}:")
                            self.output_area.append(content)
                        
                        self.output_area.append("")
                
                message_count = len([e for e in self.agent.display_history 
                                   if e.get('role') in ['user', 'assistant']])
                self.output_area.append(f"\n=== HISTORY LOADED ({message_count} messages) ===\n")
            else:
                self.output_area.append("No conversation history for this agent.\n")
                
        except Exception as e:
            self.output_area.append(f"Error loading history: {str(e)}\n")
    
    def update_title(self):
        """Update panel title"""
        if self.agent:
            self.setTitle(f"Agent: {self.agent.name}")
        else:
            self.setTitle("Agent Panel")
    
    def set_other_panel(self, other_panel):
        """Set reference to the other agent panel"""
        self.other_panel = other_panel
    
    def send_individual_message(self, message):
        """Send message to this agent only"""
        if not self.agent:
            self.output_area.append("\n❌ No agent selected\n")
            return
        
        self.output_area.append(f"\n{'='*60}")
        self.output_area.append(f"📤 You → {self.agent.name}:")
        self.output_area.append(message)
        self.output_area.append(f"{'='*60}\n")
        
        self.input_area.setEnabled(False)
        self.vulnerability_btn.setEnabled(False)
        self.reflection_btn.setEnabled(False)
        
        # Use orchestrator worker
        self.worker = AgentWorker(self.orchestrator, self.agent.name, message)
        self.worker.result_ready.connect(self.on_response_ready)
        self.worker.start()
    
    def on_response_ready(self, response):
        """Handle agent response"""
        self.output_area.append(f"💬 {self.agent.name}:")
        self.output_area.append(response)
        self.output_area.append("\n")
        
        # Re-enable inputs
        self.input_area.setEnabled(True)
        self.vulnerability_btn.setEnabled(True)
        self.reflection_btn.setEnabled(True)
    
    def on_vulnerability_clicked(self):
        """Request vulnerability analysis from the other agent"""
        if not self.other_panel or not self.other_panel.agent:
            self.output_area.append("\n❌ No other agent available for analysis\n")
            return
        
        if not self.agent.latest_response:
            self.output_area.append("\n❌ No response available to analyze\n")
            return
        
        self.output_area.append(f"\n🔍 Requesting vulnerability analysis from {self.other_panel.agent.name}...")
        
        # Disable buttons during analysis
        self.vulnerability_btn.setEnabled(False)
        self.reflection_btn.setEnabled(False)
        self.input_area.setEnabled(False)
        
        # Start vulnerability worker
        self.vulnerability_worker = VulnerabilityWorker(
            self.orchestrator,
            self.agent.name,  # source agent (this agent)
            self.other_panel.agent.name  # target agent (other agent)
        )
        self.vulnerability_worker.result_ready.connect(self.on_vulnerability_ready)
        self.vulnerability_worker.start()
    
    def on_vulnerability_ready(self, request, response):
        """Handle vulnerability analysis results"""
        # Show in the OTHER panel (where the analysis was performed)
        if self.other_panel:
            self.other_panel.output_area.append(f"\n{'='*60}")
            self.other_panel.output_area.append(f"🔍 Vulnerability Analysis Request:")
            self.other_panel.output_area.append(request[:200] + "..." if len(request) > 200 else request)
            self.other_panel.output_area.append(f"\n💬 {self.other_panel.agent.name}'s analysis:")
            self.other_panel.output_area.append(response)
            self.other_panel.output_area.append(f"{'='*60}\n")
        
        self.output_area.append("✅ Vulnerability analysis complete. Check other agent's panel.")
        
        # Re-enable buttons
        self.vulnerability_btn.setEnabled(True)
        self.reflection_btn.setEnabled(True)
        self.input_area.setEnabled(True)
    
    def on_reflection_clicked(self):
        """Request this agent to reflect on the other agent's critique"""
        if not self.other_panel or not self.other_panel.agent:
            self.output_area.append("\n❌ No other agent available\n")
            return
        
        if not self.other_panel.agent.latest_response:
            self.output_area.append("\n❌ No critique available from other agent\n")
            return
        
        self.output_area.append(f"\n🔄 Reflecting on {self.other_panel.agent.name}'s critique...")
        
        # Disable buttons during reflection
        self.vulnerability_btn.setEnabled(False)
        self.reflection_btn.setEnabled(False)
        self.input_area.setEnabled(False)
        
        # Start reflection worker
        self.reflection_worker = ReflectionWorker(
            self.orchestrator,
            self.other_panel.agent.name,  # source agent (providing critique)
            self.agent.name  # target agent (this agent, reflecting)
        )
        self.reflection_worker.result_ready.connect(self.on_reflection_ready)
        self.reflection_worker.start()
    
    def on_reflection_ready(self, request, response):
        """Handle reflection results"""
        self.output_area.append(f"📋 Reflection prompt:")
        self.output_area.append(request[:200] + "..." if len(request) > 200 else request)
        self.output_area.append(f"\n💬 {self.agent.name}'s refined response:")
        self.output_area.append(response)
        self.output_area.append("\n")
        
        # Re-enable buttons
        self.vulnerability_btn.setEnabled(True)
        self.reflection_btn.setEnabled(True)
        self.input_area.setEnabled(True)
    
    def update_fonts(self):
        """Update fonts for all elements in this panel"""
        if hasattr(self.parent_gui, 'current_font_size'):
            font_size = self.parent_gui.current_font_size
            
            # Update all widgets
            font = QFont()
            font.setPointSize(font_size)
            
            self.agent_label.setFont(font)
            self.agent_selector.setFont(font)
            self.output_area.setFont(font)
            self.input_area.setFont(font)
            self.vulnerability_btn.setFont(font)
            self.reflection_btn.setFont(font)
            
            # Update group box title font
            self.setStyleSheet(f"QGroupBox {{ font-size: {font_size}pt; font-weight: bold; }}")


class GrantReviewGUI(QWidget):
    """Main window for grant review with two agents side-by-side"""
    
    def __init__(self, config_file=None):
        super().__init__()
        
        # If no config file provided, show file dialog
        if not config_file:
            config_file, _ = QFileDialog.getOpenFileName(
                None,
                "Select Configuration File",
                "",
                "JSON files (*.json)"
            )
            if not config_file:
                QMessageBox.critical(None, "Error", "No configuration file selected")
                sys.exit(1)
        
        self.config_file = config_file
        self.orchestrator = MultiAgentOrchestrator(config_file)
        
        # Font size management
        self.current_font_size = 10
        self.min_font_size = 8
        self.max_font_size = 20
        
        # Get active agents
        active_agents = [a for a in self.orchestrator.agents if a.active]
        
        # Select first two agents by default
        self.agent1 = active_agents[0] if len(active_agents) > 0 else None
        self.agent2 = active_agents[1] if len(active_agents) > 1 else active_agents[0] if len(active_agents) > 0 else None

        # Store reference to workers to prevent garbage collection
        self.file_worker = None
        self.broadcast_worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the main UI with proportional sizing"""
        self.setWindowTitle("Two Agent Collaborative Analysis")
        self.setGeometry(100, 100, 1400, 800)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)  # Padding around edges
        
        # Top button bar
        button_bar = QHBoxLayout()
        
        # File operations buttons
        upload_btn = QPushButton("📄 Upload")
        upload_btn.clicked.connect(self.upload_file)
        button_bar.addWidget(upload_btn)
        
        load_btn = QPushButton("📁 Load Config")
        load_btn.clicked.connect(self.load_config)
        button_bar.addWidget(load_btn)
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.clear_all)
        button_bar.addWidget(clear_btn)
        
        button_bar.addStretch()
        
        # Font size controls
        font_label = QLabel("Font:")
        button_bar.addWidget(font_label)
        
        self.font_minus_btn = QPushButton("-")
        self.font_minus_btn.setMaximumWidth(30)
        self.font_minus_btn.clicked.connect(self.decrease_font)
        button_bar.addWidget(self.font_minus_btn)
        
        self.font_size_label = QLabel(str(self.current_font_size))
        self.font_size_label.setMinimumWidth(30)
        self.font_size_label.setAlignment(Qt.AlignCenter)
        button_bar.addWidget(self.font_size_label)
        
        self.font_plus_btn = QPushButton("+")
        self.font_plus_btn.setMaximumWidth(30)
        self.font_plus_btn.clicked.connect(self.increase_font)
        button_bar.addWidget(self.font_plus_btn)
        
        button_bar.addStretch()
        
        exit_btn = QPushButton("❌ Exit")
        exit_btn.clicked.connect(self.close)
        button_bar.addWidget(exit_btn)
        
        main_layout.addLayout(button_bar)
        
        # Title
        self.title_label = QLabel("Two Agent Collaborative Analysis")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_font = QFont()
        self.title_font.setPointSize(16)
        self.title_font.setBold(True)
        self.title_label.setFont(self.title_font)
        main_layout.addWidget(self.title_label)
        
        # Instructions
        self.instructions_label = QLabel(
            "Broadcast messages to both agents below. Use Vulnerability to find flaws, "
            "and Reflection to improve responses based on critique."
        )
        self.instructions_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.instructions_label)
        
        # Content area with proportional sizing
        content_splitter = QSplitter(Qt.Vertical)
        
        # Broadcast area (20% of height)
        self.broadcast_group = QGroupBox("Broadcast to Both Agents")
        broadcast_layout = QVBoxLayout()
        self.broadcast_input = BroadcastTextEdit(self)
        broadcast_layout.addWidget(self.broadcast_input)
        self.broadcast_group.setLayout(broadcast_layout)
        
        # Agent panels area (75% of height)
        agent_splitter = QSplitter(Qt.Horizontal)
        
        # Create agent panels
        self.panel1 = AgentPanel(self.agent1, self.orchestrator, 1, self)
        self.panel2 = AgentPanel(self.agent2, self.orchestrator, 2, self)
        
        # Cross-reference panels
        self.panel1.set_other_panel(self.panel2)
        self.panel2.set_other_panel(self.panel1)
        
        agent_splitter.addWidget(self.panel1)
        agent_splitter.addWidget(self.panel2)
        agent_splitter.setSizes([700, 700])
        
        # Add to content splitter with proportional sizes
        content_splitter.addWidget(self.broadcast_group)
        content_splitter.addWidget(agent_splitter)
        
        # Set initial proportions (20% broadcast, 75% agents, 5% padding implicit)
        total_height = 800 - 100  # Approximate height minus buttons and padding
        content_splitter.setSizes([int(total_height * 0.20), int(total_height * 0.75)])
        
        main_layout.addWidget(content_splitter)
        
        self.setLayout(main_layout)
        
        # Store references to widgets that need font updates
        self.all_buttons = [upload_btn, load_btn, clear_btn, exit_btn, 
                           self.font_minus_btn, self.font_plus_btn]
        self.all_labels = [font_label, self.font_size_label, 
                          self.title_label, self.instructions_label]

        # Apply initial font size
        self.update_all_fonts()        
    
    def increase_font(self):
        """Increase font size across all elements"""
        if self.current_font_size < self.max_font_size:
            self.current_font_size += 1
            self.update_all_fonts()
    
    def decrease_font(self):
        """Decrease font size across all elements"""
        if self.current_font_size > self.min_font_size:
            self.current_font_size -= 1
            self.update_all_fonts()
    
    def update_all_fonts(self):
        """Update fonts for all elements in the application"""
        font = QFont()
        font.setPointSize(self.current_font_size)
        
        # Update all buttons
        if hasattr(self, 'all_buttons'):
            for btn in self.all_buttons:
                btn.setFont(font)
        
        # Update all labels (except title which is larger)
        for label in self.all_labels:
            if label == self.title_label:
                title_font = QFont()
                title_font.setPointSize(self.current_font_size + 6)
                title_font.setBold(True)
                label.setFont(title_font)
            else:
                label.setFont(font)
        
        # Update font size display
        self.font_size_label.setText(str(self.current_font_size))
        
        # Update broadcast area
        self.broadcast_input.setFont(font)
        self.broadcast_group.setStyleSheet(f"QGroupBox {{ font-size: {self.current_font_size}pt; font-weight: bold; }}")
        
        # Update agent panels
        self.panel1.update_fonts()
        self.panel2.update_fonts()
    
    def broadcast_to_agents(self, message):
        """Broadcast message to both agents using orchestrator"""
        # Display broadcast message in both panels
        self.panel1.output_area.append(f"\n{'='*60}")
        self.panel1.output_area.append(f"📢 Broadcast → {self.panel1.agent.name if self.panel1.agent else 'Agent'}:")
        self.panel1.output_area.append(message)
        self.panel1.output_area.append(f"{'='*60}\n")
        
        self.panel2.output_area.append(f"\n{'='*60}")
        self.panel2.output_area.append(f"📢 Broadcast → {self.panel2.agent.name if self.panel2.agent else 'Agent'}:")
        self.panel2.output_area.append(message)
        self.panel2.output_area.append(f"{'='*60}\n")
        
        # Disable inputs
        self.panel1.input_area.setEnabled(False)
        self.panel1.vulnerability_btn.setEnabled(False)
        self.panel1.reflection_btn.setEnabled(False)
        
        self.panel2.input_area.setEnabled(False)
        self.panel2.vulnerability_btn.setEnabled(False)
        self.panel2.reflection_btn.setEnabled(False)
        
        # Use orchestrator's broadcast_message
        self.broadcast_worker = BroadcastWorker(self.orchestrator, message)
        self.broadcast_worker.results_ready.connect(self.on_broadcast_ready)
        self.broadcast_worker.start()
    
    def on_broadcast_ready(self, responses):
        """Handle broadcast responses"""
        # Display responses in respective panels
        if self.panel1.agent and self.panel1.agent.name in responses:
            self.panel1.output_area.append(f"💬 {self.panel1.agent.name}:")
            self.panel1.output_area.append(responses[self.panel1.agent.name])
            self.panel1.output_area.append("\n")
        
        if self.panel2.agent and self.panel2.agent.name in responses:
            self.panel2.output_area.append(f"💬 {self.panel2.agent.name}:")
            self.panel2.output_area.append(responses[self.panel2.agent.name])
            self.panel2.output_area.append("\n")
        
        # Re-enable inputs
        self.panel1.input_area.setEnabled(True)
        self.panel1.vulnerability_btn.setEnabled(True)
        self.panel1.reflection_btn.setEnabled(True)
        
        self.panel2.input_area.setEnabled(True)
        self.panel2.vulnerability_btn.setEnabled(True)
        self.panel2.reflection_btn.setEnabled(True)
    
    def upload_file(self):
        """Upload and process a file with multi-modal capabilities"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file to upload",
            "",
            "All Supported (*.pdf *.txt *.png *.jpg *.jpeg *.gif *.bmp *.mp3 *.wav *.ogg *.m4a);;"
            "PDF files (*.pdf);;"
            "Text files (*.txt);;"
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp);;"
            "Audio files (*.mp3 *.wav *.ogg *.m4a)"
        )
        
        if file_path:
            # Process file in worker thread
            self.file_worker = FileProcessorWorker(file_path)
            self.file_worker.result_ready.connect(self.on_file_processed)
            self.file_worker.start()
    
    def on_file_processed(self, content, should_broadcast):
        """Handle processed file content"""
        if should_broadcast:
            # Add content to broadcast input for user to review/edit before sending
            self.broadcast_input.setPlainText(content)
        else:
            # Just show the error/info message in both panels
            self.panel1.output_area.append(f"\n{content}")
            self.panel2.output_area.append(f"\n{content}")
    
    def load_config(self):
        """Load a configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration File",
            "",
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                # Reload orchestrator with new config
                self.orchestrator = MultiAgentOrchestrator(file_path)
                self.config_file = file_path
                
                # Update agent panels
                self.panel1.orchestrator = self.orchestrator
                self.panel2.orchestrator = self.orchestrator
                self.panel1.populate_agent_selector()
                self.panel2.populate_agent_selector()
                
                # Clear displays
                self.panel1.output_area.clear()
                self.panel2.output_area.clear()
                self.broadcast_input.clear()
                
                QMessageBox.information(self, "Success", f"Configuration loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")
    
    def clear_all(self):
        """Clear all conversation history"""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "This will clear all conversation history. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset through orchestrator
            self.orchestrator.reset_all_agents()
            
            # Clear displays
            self.panel1.output_area.clear()
            self.panel2.output_area.clear()
            self.broadcast_input.clear()
            
            # Show reset messages
            self.panel1.output_area.append("✅ Agent reset complete")
            self.panel2.output_area.append("✅ Agent reset complete")


def main():
    """Main entry point for grant review GUI"""
    app = QApplication(sys.argv)
    
    # Check if config file is provided as argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        window = GrantReviewGUI(config_file)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to start: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()