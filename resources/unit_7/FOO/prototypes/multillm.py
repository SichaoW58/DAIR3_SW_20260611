import tkinter as tk
from tkinter import ttk
from Agent import Agent
from tqdm import tqdm
import json
import random
import string
import os

class ChatInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat Interface")

        with open('config.json', 'r') as f:
            config = json.load(f)

        self.instructions = config['CONFIG']['instructions']
        self.name = config['CONFIG']['name']
        self.user = config['CONFIG']['user']

        self.ensemble_agent = Agent(instruction_set=self.instructions)
        self.consensus_agent = Agent(instruction_set=self.instructions)

        self.model_counts = {"OpenAI": 0, "Gemini": 0, "Claude": 0}

        # Input fields
        self.inputs = {}
        input_labels = ["Original Prompt", "Revision Prompt", "Number of Iterations"]
        for i, label_text in enumerate(input_labels):
            label = ttk.Label(root, text=label_text)
            label.grid(row=i, column=0, sticky='e', padx=5, pady=5)
            entry = ttk.Entry(root, width=40)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.inputs[label_text] = entry

        # Bind Enter to message input
        self.inputs["Number of Iterations"].bind('<Return>', self.on_chat)

        # Chat Button
        self.chat_button = ttk.Button(root, text="Chat", command=self.on_chat)
        self.chat_button.grid(row=3, column=1, pady=10, sticky='e')

        # Writing Ensemble Model Buttons Frame
        self.ensemble_frame = ttk.LabelFrame(root, text="Add Writing Ensemble Model")
        self.ensemble_frame.grid(row=3, column=0, padx=5, pady=5, sticky='w')

        ttk.Button(self.ensemble_frame, text="Add OpenAI", command=lambda: self.add_ensemble("OpenAI")).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.ensemble_frame, text="Add Gemini", command=lambda: self.add_ensemble("Gemini")).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(self.ensemble_frame, text="Add Claude", command=lambda: self.add_ensemble("Claude")).grid(row=0, column=2, padx=5, pady=2)

        # Consensus Builder Model Buttons Frame
        self.consensus_frame = ttk.LabelFrame(root, text="Add Consensus Builder Model")
        self.consensus_frame.grid(row=4, column=0, padx=5, pady=5, sticky='w')

        ttk.Button(self.consensus_frame, text="Add OpenAI", command=lambda: self.add_consensus("OpenAI")).grid(row=0, column=0, padx=5, pady=2)
        ttk.Button(self.consensus_frame, text="Add Gemini", command=lambda: self.add_consensus("Gemini")).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(self.consensus_frame, text="Add Claude", command=lambda: self.add_consensus("Claude")).grid(row=0, column=2, padx=5, pady=2)

        # Output display
        self.output_text = tk.Text(root, height=20, width=100, state='disabled')
        self.output_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    def save_history_to_file(self, history):
        # Create 'history' directory if it doesn't exist
        history_dir = 'history'
        os.makedirs(history_dir, exist_ok=True)

        # Generate random filename
        filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.txt'
        filepath = os.path.join(history_dir, filename)

        # Write history to file
        with open(filepath, 'w') as f:
            for item in history:
                f.write(f"TYPE: {item['type']}\n")
                f.write(f"NAME: {item['name']}\n")
                f.write(f"ITERATION: {item['iteration']}\n")
                f.write(f"PROMPT:\n{item['prompt']}\n\n")
                f.write(f"RESPONSE:\n{item['response']}\n")
                f.write("-" * 50 + "\n\n")

        self.append_output(f"[System] Conversation history saved to {filepath}\n")

    def on_chat(self, event=None):
        prompt = self.inputs["Original Prompt"].get()
        revision_prompt = self.inputs["Revision Prompt"].get()
        try:
            iterations = int(self.inputs["Number of Iterations"].get())
        except ValueError:
            self.append_output("[Error] Number of iterations must be an integer\n")
            return

        self.append_output(f"\nChatting with: multi-agent system\n{'=' * 20}\n")

        history = []
        responses = {}
        initial_responses = {}

        # Initial generation
        for name in self.ensemble_agent.clients:
            response = self.ensemble_agent.generate(name, prompt)
            initial_responses[name] = response
            history.append({
                'type': 'initial',
                'name': name,
                'iteration': 0,
                'prompt': prompt,
                'response': response
            })

        responses = initial_responses.copy()

        for iteration in tqdm(range(iterations), desc="Feedback/Revision Iterations"):
            feedback = {name: [] for name in self.ensemble_agent.clients}

            for reviewer in self.ensemble_agent.clients:
                for target in self.ensemble_agent.clients:
                    if reviewer == target:
                        continue
                    feedback_prompt = f"Here is a peer response to the prompt:\n\n{responses[target]}\n\n{revision_prompt}"
                    fb = self.ensemble_agent.generate(reviewer, feedback_prompt)
                    feedback[target].append(f"{reviewer} says:\n{fb.strip()}")
                    history.append({
                        'type': 'feedback',
                        'name': reviewer,
                        'target': target,
                        'iteration': iteration + 1,
                        'prompt': feedback_prompt,
                        'response': fb.strip()
                    })

            new_responses = {}
            for name in self.ensemble_agent.clients:
                revision_input = (
                    f"The original prompt was:\n{prompt}\n\n"
                    f"Your previous response was:\n{responses[name]}\n\n"
                    f"Here is feedback from your peers:\n\n" +
                    "\n\n".join(feedback[name]) +
                    "\n\nPlease revise your response based on this feedback."
                )
                revised = self.ensemble_agent.generate(name, revision_input)
                new_responses[name] = revised
                history.append({
                    'type': 'revision',
                    'name': name,
                    'iteration': iteration + 1,
                    'prompt': revision_input,
                    'response': revised
                })

            responses = new_responses

        synthesis_input = (
            f"The original prompt was:\n{prompt}\n\n"
            f"Here are the final revised responses from each agent:\n\n" +
            "\n\n".join([f"{name}:\n{resp}" for name, resp in responses.items()]) +
            "\n\nPlease synthesize the strongest final response using the best elements of each. Clean up the text into a single concise passage (4-6 sentences). Make sure it is written as scientifically as possible. DO NOT use any adjectives of participle phrases."
        )

        final_model = next(iter(self.consensus_agent.clients))
        final_response = self.consensus_agent.generate(final_model, synthesis_input)
        self.append_output(f"\nFinal Consensus:\n{final_response}\n{'=' * 40}\n")
        history.append({
            'type': 'synthesis',
            'name': final_model,
            'iteration': iterations + 1,
            'prompt': synthesis_input,
            'response': final_response
        })

        self.save_history_to_file(history)


    def append_output(self, text):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, text)
        self.output_text.config(state='disabled')
        self.output_text.see(tk.END)

    def add_ensemble(self, model):
        count = self.model_counts[model]
        model_id = f"{model.lower()}_{count}"
        if model == "OpenAI":
            self.ensemble_agent.add_openai(name=model_id)
        elif model == "Gemini":
            self.ensemble_agent.add_gemini(name=model_id)
        elif model == "Claude":
            self.ensemble_agent.add_anthropic(name=model_id)
        self.model_counts[model] += 1
        self.append_output(f"[System] Added {model} to Ensemble as {model_id}\n")

    def add_consensus(self, model):
        model_id = f"{model.lower()}_{self.model_counts[model]}"
        if model == "OpenAI":
            self.consensus_agent.add_openai(name=model_id)
        elif model == "Gemini":
            self.consensus_agent.add_gemini(name=model_id)
        elif model == "Claude":
            self.consensus_agent.add_anthropic(name=model_id)
        self.append_output(f"[System] Added {model} to Consensus as {model_id}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatInterface(root)
    root.mainloop()
