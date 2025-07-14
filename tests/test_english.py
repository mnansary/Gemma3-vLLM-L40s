import openai

# Configure the OpenAI client to connect to your local server
client = openai.OpenAI(
    api_key="YOUR_SUPER_SECRET_KEY",
    base_url="http://localhost:24434/v1",
)

# Define the conversation
messages = [
    {"role": "user", "content": "What are the top 3 benefits of using a GPU for deep learning?"}
]

# Create the chat completion request with streaming enabled
stream = client.chat.completions.create(
    model="RedHatAI/gemma-3-27b-it-FP8-dynamic",
    messages=messages,
    max_tokens=300,
    stream=True,
)

# Print each chunk of the response as it is received
print("Assistant's Response (streaming):")
for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print("\n--- End of Stream ---")

