import os
import json
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OpenAI API key not found. Set the OPENAI_API_KEY environment variable "
        "or create a .env file with OPENAI_API_KEY=your_api_key"
    )

client = OpenAI(api_key=api_key)

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def search_movie_database(query_type, search_term):
    """
    Search the database for movie or actor
    query_type: "actor", "movie"
    search_term: Name to search for
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor(cursor_factory=DictCursor)

        if query_type == "actor":
            cursor.execute(
                "SELECT actor.actor_id, actor.first_name, actor.last_name, film.title FROM actor " +
                "JOIN film_actor ON actor.actor_id = film_actor.actor_id " +
                "JOIN film ON film_actor.film_id = film.film_id " +
                "WHERE actor.first_name ILIKE %s OR actor.last_name ILIKE %s",
                (f"%{search_term}%", f"%{search_term}%")
            )
        elif query_type == "movie":
            cursor.execute(
                "SELECT film.film_id, film.title, film.description, film.release_year, " +
                "film.rental_rate, film.rating, film.length FROM film WHERE title ILIKE %s",
                (f"%{search_term}%",)
            )

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        if results:
            return {
                "success": True,
                "data": [dict(row) for row in results],
                "message": f"Found {len(results)} results for {query_type} '{search_term}'"
            }
        else:
            return {
                "success": False,
                "data": [],
                "message": f"No {query_type} found matching '{search_term}'"
            }
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "message": f"Database error: {str(e)}"
        }


def extract_query_info(message):
    """
    Use the LLM to extract query type and search term from user message
    """
    system_prompt = """You are a helpful assistant that extracts query information from user messages about movies.
    Your task is to analyze the user message and identify if they are asking about:
    1. A specific movie (query_type: "movie")
    2. A specific actor (query_type: "actor")

    If the message contains a query, extract the search term (movie title or actor name).
    If no specific query is detected, return query_type as null and search_term as null.

    Return only a JSON object with the following structure:
    {"query_type": "movie|actor|null", "search_term": "extracted term or null"}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error extracting query info: {str(e)}")
        return {"query_type": None, "search_term": None}


def format_db_results(results, query_type):
    """Format database results into readable text for the LLM"""
    if not results["success"] or not results["data"]:
        return results["message"]

    formatted_text = ""

    if query_type == "movie":
        formatted_text = "I found the following movies in our database:\n\n"
        for movie in results["data"]:
            formatted_text += f"Title: {movie['title']}\n"
            formatted_text += f"Release Year: {movie['release_year']}\n"
            formatted_text += f"Description: {movie['description']}\n"
            formatted_text += f"Rating: {movie['rating']}\n"
            formatted_text += f"Duration: {movie['length']} minutes\n"
            formatted_text += f"Rental Rate: ${movie['rental_rate']}\n\n"

    elif query_type == "actor":
        formatted_text = "I found the following actor appearances in our database:\n\n"
        for role in results["data"]:
            formatted_text += f"Actor: {role['first_name']} {role['last_name']}\n"
            formatted_text += f"Appeared in: {role['title']}\n\n"

    return formatted_text


def chat_with_openai(message, history):
    system_prompt = """You are a film expert agent who has worked in the world's most prestigious video rental stores.
    You know all the details of every movie: from the cast and director to duration, rating, and genre.
    As a critic who has watched all movies, you have a well-founded opinion on each one.
    When receiving questions about any movie's synopsis, you will also provide an evaluation based on technical and content criteria.
    You will respond kindly, but will limit yourself to talking exclusively about movies.

    When I provide database information, treat it as authoritative. If specific information was found in our database,
    incorporate it into your answer and clearly indicate it's from "our VideoClub database." 

    Format the database information to make it readable and engaging, but don't fabricate additional database details.
    For information not explicitly found in the database, you may supplement with your general knowledge but always
    indicate when you're doing so by saying phrases like "Based on my knowledge..." or "While our database doesn't 
    specify this, I can tell you that..."

    Maintain a conversational, friendly tone, as if you're a passionate film expert helping a customer at the video store.
    """

    messages = [{"role": "system", "content": system_prompt}]

    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    query_info = extract_query_info(message)
    query_type = query_info.get("query_type")
    search_term = query_info.get("search_term")

    print(f"Extracted query - Type: {query_type}, Term: {search_term}")  # Debug info

    db_results = None
    if query_type and search_term:
        db_results = search_movie_database(query_type, search_term)
        if db_results["success"]:
            formatted_results = format_db_results(db_results, query_type)
            augmented_message = (
                    message +
                    "\n\n[DATABASE RESULTS FOR ASSISTANT REFERENCE (NOT VISIBLE TO USER):\n" +
                    formatted_results +
                    "\nPlease incorporate this database information into your response naturally.]"
            )
        else:
            augmented_message = (
                    message +
                    "\n\n[DATABASE NOTE FOR ASSISTANT REFERENCE (NOT VISIBLE TO USER): " +
                    db_results["message"] +
                    ". Please mention that this specific information wasn't found in our database, " +
                    "but provide your general knowledge about the topic.]"
            )
    else:
        augmented_message = message

    messages.append({"role": "user", "content": augmented_message})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1,
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.ChatInterface(
    chat_with_openai,
    title="VideoClub Agent • ESESA",
    description="Your personal movie expert. Ask me about any films, actors, or cinema topics!"
                + " • Database content is in English, please use English as well in your conversation.",
)

if __name__ == "__main__":
    demo.launch(share=True)
