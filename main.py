import os
import warnings
import logging
import librosa
import numpy as np
import speech_recognition as sr
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pydub import AudioSegment

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

# Load emotion model
try:
    model = load_model("speech_emotion_recognition_model.keras")
except Exception as e:
    print(f" Failed to load emotion model: {e}")
    exit()

# Recording and transcription using sr
def recording():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎙 Listening... Speak something:")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            print(" Processing...")

            # Transcribe the speech
            try:
                text = recognizer.recognize_google(audio)
                print(" Transcription:", text)
            except sr.UnknownValueError:
                print(" Could not understand the audio.")
                text = ""
            except sr.RequestError as e:
                print(f"⚠ Could not request results; {e}")
                text = ""

            # Save audio to .wav
            wav_path = "recorded_audio.wav"
            with open(wav_path, "wb") as f:
                f.write(audio.get_wav_data())
            print(" Audio saved as", wav_path)

            # Convert to .mp3
            try:
                mp3_path = "audio.mp3"
                sound = AudioSegment.from_wav(wav_path)
                sound.export(mp3_path, format="mp3")
                print(" Converted to MP3:", mp3_path)
            except Exception as e:
                print(f" Error converting to MP3: {e}")
                return text, None

            return text, mp3_path

    except Exception as e:
        print(f" Error during recording: {e}")
        return "", None


# Feature extraction
def extract_features(data, sample_rate=22050):
    try:
        result = np.array([])
        result = np.hstack((result, np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)))
        stft = np.abs(librosa.stft(data))
        result = np.hstack((result, np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)))
        result = np.hstack((result, np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)))
        result = np.hstack((result, np.mean(librosa.feature.rms(y=data).T, axis=0)))
        result = np.hstack((result, np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)))
        return result
    except Exception as e:
        print(f" Feature extraction failed: {e}")
        return None


# Emotion prediction
def model_prediction(audio_path):
    try:
        categories = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        encoder = LabelEncoder()
        encoder.fit(categories)

        y, sr = librosa.load(audio_path)
        features = extract_features(y, sample_rate=sr)

        if features is None:
            raise ValueError("Features are None")

        reshaped = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
        prediction = model.predict(reshaped)
        index = np.argmax(prediction)
        predicted_emotion = encoder.classes_[index]

        print(" Model Emotion Prediction:", predicted_emotion)
        return predicted_emotion

    except Exception as e:
        print(f" Emotion prediction failed: {e}")
        return "unknown"


# LangChain LLM setup
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

prompt = PromptTemplate(
    input_variables=["sentence", "prediction"],
    template="""
You are an expert in emotion and mood detection.
Given a sentence and a model's prediction, evaluate the emotion **accurately**, especially when the model might be wrong. Return:
- One-word **Emotion**
- One-word **Mood**

Sentence: {sentence}

Model Prediction:{prediction}

Instruction: There is a high chance the prediction is incorrect. You must analyze the sentence and determine the correct emotion and mood.

Respond in the following format:
Emotion: <one word>
Mood: <one word>
"""
)

emotion_chain = LLMChain(llm=llm, prompt=prompt)

# --- Main Flow ---
try:
    text, audio_path = recording()
    if not audio_path:
        print(" Recording failed. Exiting.")
        exit()

    predicted_emotion = model_prediction(audio_path)

    try:
        result = emotion_chain.run(sentence=text, prediction=predicted_emotion)
    except Exception as e:
        print(f" LLM response failed: {e}")
        result = "Emotion: unknown\nMood: unknown"

    print("\n--- Final Output ---")
    print("🎙 Input Sentence:", text)
    print(" Model Prediction:", predicted_emotion)
    print(" LLM Refined Output:\n", result)

except Exception as e:
    print(f" Unexpected error in execution: {e}")
