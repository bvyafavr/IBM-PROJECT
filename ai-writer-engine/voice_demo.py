import os
import speech_recognition as sr
import pyttsx3
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load Gemini API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    google_api_key=gemini_api_key,
    temperature=0.7
)

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts = pyttsx3.init()

def speak(text):
    print("AI:", text)
    tts.say(text)
    tts.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        print("You:", query)
        return query
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"⚠️ Speech recognition error: {e}")
        return None

# Main loop
while True:
    user_input = listen()
    if user_input is None:
        continue
    if user_input.lower() in ["exit", "quit", "bye"]:
        speak("Goodbye!")
        break
    response = llm.invoke(user_input)
    speak(response.content)
