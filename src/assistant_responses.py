import os
import openai
import pyttsx3
from openai import OpenAI
from database import get_session_interactions
from sqlite_vec import serialize_float32
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()


class AssistantResponder:
    """
    A class that uses OpenAI to generate responses and also provides
    text-to-speech capabilities for speaking the generated response.
    """

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.engine = pyttsx3.init()  # Initializes the TTS engine

        # Optionally set engine properties (voice, rate, volume, etc.)
        # Example:
        # self.engine.setProperty('rate', 175)

    def get_response(self, session_id: int) -> str:
        
        """
        Generate a response from OpenAI using all previous user and assistant
        messages in the current session as the conversation context.
        """
        db_session = self.session_factory()
        try:
            # 1) Get all interactions for the session
            interactions = get_session_interactions(self.session_factory, session_id)

            # 2) Build a conversation context list of messages
            #    We add a system prompt at the beginning to set the tone or instructions of the assistant
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant that specializes in helping users plan events. "
                        "You have access to the conversation so far. Respond in a concise, polite, and helpful way."
                    )
                }
            ]

            # Sort interactions by timestamp (just in case they aren't sorted).
            interactions_sorted = sorted(interactions, key=lambda x: x.timestamp)

            # Convert each interaction to the appropriate role/content for the chat
            for interaction in interactions_sorted:
                if interaction.role == "assistant":
                    messages.append({"role": "assistant", "content": interaction.transcript})
                else:
                    # We'll treat "user" as a normal user message
                    messages.append({"role": "user", "content": interaction.transcript})

            # 3) Call the OpenAI Chat Completion endpoint
            openai_client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.7, stream=False)
            
            # 4) Extract the assistant's text from the response
            assistant_text = response.choices[0].message.content

            return assistant_text

        except Exception as e:
            print(f"Error in generate_openai_response: {e}")
            return "I'm sorry, but I ran into an error. Could you please try again?"
        finally:
            db_session.close()


    def speak_response(self, text: str):
        """
        Convert the given text to speech using pyttsx3.
        """
        if not text:
            return  # Don't speak empty text
        self.engine.say(text)
        self.engine.runAndWait()
