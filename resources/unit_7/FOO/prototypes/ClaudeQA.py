"""
ClaudeQA.py (prototype)
Single-shot Claude question-answer demo; kept for reference.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import anthropic
import os

# Create the client using the API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    temperature=0.5,
    messages=[
        {
            "role": "user",
            "content": "Is it true that Emperor Nero declared war on the sea?"
        }
    ]
)

text_blocks = message.content 

# Extract and print the text from the first TextBlock
for block in text_blocks:
    print(block.text)

# print(message.content)