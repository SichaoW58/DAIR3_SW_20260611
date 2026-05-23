import os
import anthropic

# Initialize the Anthropic client
client = anthropic.Anthropic(
    # Set your API key from environment variables
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Make an API call to create a message
message = client.messages.create(
    model="claude-3-7-sonnet-20250219",  # Or another suitable model
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ],
)

# Access the request ID directly from the message object
request_id = message._request_id  # Corrected line to access request ID

# Print the request ID
print(f"Request ID: {request_id}")

# Print the generated message content
print(message.content)
