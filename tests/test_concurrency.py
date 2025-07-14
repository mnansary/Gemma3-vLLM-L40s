import asyncio
import openai
import time

# Configure the an Asynchronous OpenAI client to connect to your local server
# Using AsyncOpenAI is crucial for concurrent requests.
client = openai.AsyncOpenAI(
    api_key="YOUR_SUPER_SECRET_KEY",
    base_url="http://localhost:24434/v1",
)

# A list of different prompts to send simultaneously
PROMPTS = [
    "Write a short, futuristic story about a programmer and their AI assistant.",
    "Explain the concept of quantum entanglement like I'm five.",
    "Create a detailed recipe for a classic Italian lasagna.",
    "Compose a haiku about a rainy day in a bustling city.",
]

async def send_request(request_id: int, prompt: str):
    """
    Sends a single streaming request and prints the response chunks.
    """
    print(f"[Request {request_id}] Starting with prompt: '{prompt[:30]}...'")
    try:
        # Define the conversation for this specific request
        messages = [{"role": "user", "content": prompt}]

        # Create the chat completion request with streaming enabled
        # Note the 'await' keyword for this asynchronous call
        stream = await client.chat.completions.create(
            model="RedHatAI/gemma-3-27b-it-FP8-dynamic",
            messages=messages,
            max_tokens=250,
            temperature=0.8,
            stream=True,
        )

        # Asynchronously iterate over the stream and print each chunk
        # 'async for' is the asynchronous version of a for-loop
        full_response = []
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response.append(content)
                # Printing each chunk as it arrives
                # print(f"[Req {request_id}] {content}", end="", flush=True) # Uncomment for char-by-char streaming view

        # Print the full assembled response for clarity
        print(f"\n--- [Request {request_id}] Final Response ---")
        print("".join(full_response))
        print(f"--- [Request {request_id}] End of Stream ---\n")

    except Exception as e:
        print(f"An error occurred in Request {request_id}: {e}")

async def main():
    """
    The main function that creates and runs all concurrent tasks.
    """
    print("--- Starting 4 concurrent streaming tests ---")
    start_time = time.time()

    # Create a list of tasks to run concurrently.
    # asyncio.create_task() schedules the coroutine to run on the event loop.
    tasks = []
    for i, prompt in enumerate(PROMPTS):
        task = asyncio.create_task(send_request(request_id=i + 1, prompt=prompt))
        tasks.append(task)

    # asyncio.gather() runs all the tasks in the list concurrently.
    # The 'main' function will pause here until all tasks are complete.
    await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"--- All 4 concurrent requests finished in {end_time - start_time:.2f} seconds ---")


# This is the standard way to run the main async function.
if __name__ == "__main__":
    asyncio.run(main())