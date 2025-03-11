import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_API_KEY")

if not api_key:
    raise ValueError(
        "OpenAI API key not found. Set the OPENAI_API_KEY environment variable "
        "or create a .env file with OPENAI_API_KEY=your_api_key"
    )

client = OpenAI(api_key=api_key)


def chat_with_openai(message, history):
    messages = [{"role": "system",
                 "content": "You are a film expert agent who has worked in the world's most prestigious video rental stores. You know all the details of every movie: from the cast and director to duration, rating, and genre. As a critic who has watched all movies, you have a well-founded opinion on each one. When receiving questions about any movie's synopsis, you will also provide an evaluation based on technical and content criteria. You will respond kindly, but will limit yourself to talking exclusively about movies. When asked about the cast or actors of a movie in your database, you will perform a search and share the data in the chat, prioritizing database information. If you don't have specific information, you will use your general knowledge, always indicating the source of the data (whether it's from the database or elsewhere)."}]

    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1.0,
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.ChatInterface(
    chat_with_openai,
    title="VideoClub Agent • ESESA flavour",
    description="This is an chat for ask questions about movies, series, actors, etc in the VideoClub",
)

if __name__ == "__main__":
    demo.launch()
