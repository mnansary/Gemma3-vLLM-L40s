import openai
import base64

# Configure the OpenAI client to connect to your local server
client = openai.OpenAI(
    api_key="YOUR_SUPER_SECRET_KEY",
    base_url="http://localhost:24434/v1",
)

# --- Image Preparation ---
# 1. Define the path to your local image
image_path = "test_image.jpg"
print(f"Reading local image from: {image_path}")

# 2. Open the image file in binary mode and encode it to base64
try:
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
except FileNotFoundError:
    print(f"ERROR: The file '{image_path}' was not found.")
    print("Please run the curl command to download it first and ensure it's in the same directory.")
    exit()

# --- API Request ---
# Define the multi-modal message payload, which remains the same
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail. What is the animal doing?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }
        ]
    }
]

# Create the chat completion request
print("\nSending image and text to the model...")
response = client.chat.completions.create(
    model="RedHatAI/gemma-3-27b-it-FP8-dynamic",
    messages=messages,
    max_tokens=200,
)

# Print the model's response
print("\n--- Model's Response ---")
print(response.choices[0].message.content)