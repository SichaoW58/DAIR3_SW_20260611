import os
from openai import OpenAI
import anthropic
from google import genai
from google.genai import types

class Agent:
    def __init__(self, instruction_set, token_window=4096):
        self.instruction_set = instruction_set
        self.max_token = token_window
        self.clients = {}  # name -> {"client", "type", "model"}
        self.consensus = {}

    def add_openai(self, name, model="gpt-4", api_key=None):
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.clients[name] = {"client": client, "type": "openai", "model": model}

    def add_anthropic(self, name, model="claude-3-opus-20240229", api_key=None):
        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.clients[name] = {"client": client, "type": "anthropic", "model": model}

    def add_gemini(self, name, model="gemini-2.5-flash-lite-preview-06-17", api_key=None):
        client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.clients[name] = {"client": client, "type": "gemini", "model": model}

    def generate(self, name, prompt):
        if name not in self.clients:
            raise ValueError(f"Model '{name}' not found")
        entry = self.clients[name]

        if entry["type"] == "openai":
            response = entry["client"].chat.completions.create(
                model=entry["model"],
                messages=[
                    {"role": "system", "content": 'Follow this instruction set strictly!! Also make sure that you do not use adjectives or participle phrases.\n\n' + self.instruction_set},
                    {"role": "user", "content": prompt}
                ], 
                max_tokens=self.max_token
            )
            return response.choices[0].message.content

        elif entry["type"] == "anthropic":
            response = entry["client"].messages.create(
                model=entry["model"],
                max_tokens=self.max_token,
                messages=[
                    {"role": "user", "content": f"Follow this instruction set strictly!! Also make sure that you do not use adjectives or participle phrases.\n\n{self.instruction_set}\n\n{prompt}"}
                ]
            )
            return response.content[0].text

        elif entry["type"] == "gemini":
            response = entry["client"].models.generate_content(
                model=entry["model"], 
                contents=['Follow this instruction set strictly!! Also make sure that you do not use adjectives or participle phrases.\n\n', self.instruction_set, prompt], 
                config=types.GenerateContentConfig(temperature=0.7, topP=0.95, maxOutputTokens=self.max_token))
            return response.text

        else:
            raise ValueError(f"Unsupported model type: {entry['type']}")