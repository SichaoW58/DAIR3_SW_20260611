"""
clsAnthropic.py
Anthropic Claude Agent class for multi-agent chat system.
Compatible with ClaudeChat.py and ClaudeGUI.py architecture.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""

import os
import anthropic
import json
import sys
import uuid
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from cls_blockchain import IntegrityManager


class ClaudeWorker(QThread):
    """Worker thread for Claude API calls to prevent GUI blocking"""
    result_ready = pyqtSignal(str)

    def __init__(self, user_input, client, model, history):
        super().__init__()
        self.user_input = user_input
        self.client = client
        self.model = model
        self.history = history

    def run(self):
        try:
            # Clean history to remove timestamps before sending to API
            clean_history = []
            for entry in self.history:
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    clean_entry = {
                        "role": entry["role"],
                        "content": entry["content"]
                    }
                    clean_history.append(clean_entry)
            
            # Add current user input
            clean_history.append({"role": "user", "content": self.user_input})
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100000,
                temperature=0.99,
                messages=clean_history
            )
            content = response.content[0].text
            
            # Add response with timestamp to the original history
            timestamp = datetime.now().isoformat()
            self.history.append({
                "role": "assistant", 
                "content": content,
                "timestamp": timestamp
            })
            
            self.result_ready.emit(content)
        except Exception as e:
            self.result_ready.emit(f"Error: {e}")


class AnthropicAgent:
    """Anthropic Claude Agent class compatible with ClaudeChat.py architecture"""
    
    def __init__(self, model, name, instructions, user, config, model_entry=None):
        self.model = model
        self.name = name
        self.user = user
        self.config = config
        self.latest_response = ""
        self.active = True
        self.integrity_issues = []
        self.integrity_valid = True        

        # Build instructions with preamble
        preamble = f"Address the user as Dr. {user}.\n\n Introduce yourself as {name}, AI assistant.\n\n "
        
        # Add agent-specific directive if available
        agent_directive = ""
        if model_entry and "agent_directive" in model_entry:
            agent_directive = f"\n\nAgent specific instructions:\n{model_entry['agent_directive']}\n"
        
        self.instructions = preamble + instructions + agent_directive
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Claude agents will not function.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Initialize conversation history
        self.history = []
        self.history.append({"role": "user", "content": self.instructions})
        self.display_history = []  # For UI display with timestamps
        
        # Set up history file path using CWD from config
        cwd = config.get("CWD", "/chats")
        if cwd.startswith("/"):
            cwd_path = cwd[1:]  # Remove leading slash for relative path
        else:
            cwd_path = cwd
            
        self.history_file = os.path.join(cwd_path, f"{self.name}.json")
        print(f"Anthropic Agent {self.name} will use history file: {self.history_file}")
        
        # Ensure the directory exists
        os.makedirs(cwd_path, exist_ok=True)
        
        # History tracking
        self.history_data = {"history": self.history, "seeded": True, "chat_id": None}
        
        # Load latest conversation
        self.load_latest_conversation()

    def send_message(self, message):
        """
        Send a message to Claude.
        Returns the response or error message.
        Note: The orchestrator handles ALL blockchain integrity - both user and assistant messages.
        This method should NOT add anything to history.
        """
        try:
            # Get current history for API call (clean for API)
            clean_history = []
            for entry in self.history:
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    clean_entry = {
                        "role": entry["role"],
                        "content": entry["content"]
                    }
                    clean_history.append(clean_entry)
            
            # Add the current user message for API call
            clean_history.append({"role": "user", "content": message})
            
            # Send to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.99,
                messages=clean_history
            )
            content = response.content[0].text
            
            # DO NOT add anything to history here - orchestrator handles this
            # Just update latest_response for copy functionality
            self.latest_response = content
            return content
            
        except Exception as e:
            return f"Error: {e}"

    def create_worker(self, user_input):
        """Create a worker thread for GUI use"""
        return ClaudeWorker(user_input, self.client, self.model, self.history)

    def load_latest_conversation(self):
        """Load the latest conversation if it exists"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                
                if isinstance(saved_data, dict) and 'history' in saved_data:
                    history = saved_data.get('history', [])
                    if len(history) > 1:  # More than just the initial system message
                        print(f"Loading latest conversation for {self.name}")
                        self.restore_conversation_from_history(saved_data)
                        return True
            except Exception as e:
                print(f"Error loading latest conversation for {self.name}: {e}")
        return False

    def restore_conversation_from_history(self, saved_data):
        """Restore conversation from saved history data"""
        history = saved_data.get('history', [])
        chat_id = saved_data.get('chat_id', None)
        seeded = saved_data.get('seeded', False)
        
        # Clean the conversation history for API compatibility and add missing timestamps
        self.history.clear()
        self.display_history = []
        current_time = datetime.now().isoformat()
        
        for entry in history:
            if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                # Create clean entry for API calls (no timestamps)
                clean_entry = {
                    "role": entry["role"],
                    "content": entry["content"]
                }
                self.history.append(clean_entry)
                
                # Create display entry with timestamp (add if missing)
                display_entry = dict(entry)  # Copy the original entry
                if 'timestamp' not in display_entry or not display_entry['timestamp']:
                    display_entry['timestamp'] = current_time
                    print(f"Added missing timestamp to {display_entry['role']} message for {self.name}")
                
                self.display_history.append(display_entry)
        
        # Generate chat ID if missing
        if not chat_id:
            chat_id = str(uuid.uuid4())
            print(f"Generated new chat ID for {self.name}: {chat_id}")
        
        self.history_data = {
            "history": self.display_history,  # Save full history with timestamps
            "seeded": seeded,
            "chat_id": chat_id
        }
        
        # Save the updated history with timestamps and chat ID
        self.save_conversation()

    def save_conversation(self):
        """Save the current conversation to file"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write chat log for {self.name}: {e}")

    def reset_conversation(self):
        """Reset the conversation"""
        self.history.clear()
        self.history.append({"role": "user", "content": self.instructions})
        self.display_history = []
        self.history_data = {"history": self.history, "seeded": True, "chat_id": None}
        self.latest_response = ""
        self.save_conversation()
        print(f"Conversation reset for {self.name}")

    def get_info(self):
        """Get agent information"""
        return {
            "name": self.name,
            "model": self.model,
            "chat_id": self.history_data.get("chat_id"),
            "active": self.active,
            "message_count": len(self.history)
        }

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file (compatible with ClaudeGUI.py)"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            return "Error: PyPDF2 not installed. Please install it to process PDF files."
        except Exception as e:
            return f"Error extracting PDF text: {e}"

    def process_file_upload(self, file_path):
        """Process file upload (compatible with ClaudeGUI.py drag-and-drop)"""
        try:
            if file_path.lower().endswith('.pdf'):
                pdf_text = self.extract_text_from_pdf(file_path)
                message = f"I've uploaded a PDF file. Here's the content:\n\n{pdf_text}\n\nPlease analyze this PDF content."
                return self.send_message(message)
            else:
                return "Error: Only PDF files are currently supported."
        except Exception as e:
            return f"Error processing file: {e}"
        
    def get_integrity_display_text(self):
        """Get text to display integrity issues in GUI"""
        if not hasattr(self, 'integrity_valid') or self.integrity_valid:
            return ""
        
        if hasattr(self, 'integrity_issues') and self.integrity_issues:
            warning_text = "⚠️ LOG TAMPERED. TRUST HAS BEEN BREACHED. BLOCKCHAIN FAILS\n"
            warning_text += "Integrity Issues:\n"
            for issue in self.integrity_issues:
                warning_text += f"- {issue}\n"
            return warning_text
        
        return "⚠️ INTEGRITY STATUS UNKNOWN"
