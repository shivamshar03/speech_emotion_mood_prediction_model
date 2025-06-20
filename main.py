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
import whisper


load_dotenv()

# Load the model
model = load_model("speech_emotion_recognition_model.keras")
# audio_path = "/content/angry.mp3"
sst_model = whisper.load_model("small")


def recording():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("🎤 Listening... Speak something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        print("🔁 Processing...")

        with open("recorded_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())
        print("💾 Audio saved as recorded_audio.wav")

        # Convert WAV to MP3
        sound = AudioSegment.from_wav("recorded_audio.wav")
        sound.export("audio.mp3", format="mp3")
        print("✅ Audio converted and saved as audio.mp3")
        audio_path = "audio.mp3"

        try:
            # Transcribe using Google's speech recognition
            text = recognizer.recognize_google(audio)
            print("📝 Transcription: " + text)
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
        except sr.RequestError:
            print("⚠️ Could not request results from the speech recognition service.")
        return text , audio_path



def extract_features(data, sample_rate=22050):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result


def model_prediction(audio_path):
    emotion_categories = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    encoder = LabelEncoder()
    encoder.fit(emotion_categories)
    y, sr = librosa.load(audio_path)
    sample_rate = sr

    features = extract_features(y, sample_rate=sample_rate)

    reshaped_features = np.expand_dims(np.expand_dims(features, axis=0), axis=2)

    # Make the prediction
    prediction = model.predict(reshaped_features)

    # Get the predicted emotion label
    predicted_emotion_index = np.argmax(prediction)
    emotion_categories = encoder.classes_ # Get the list of emotion labels from the encoder

    predicted_emotion = emotion_categories[predicted_emotion_index]

    print(f"The predicted emotion for {audio_path} is: {predicted_emotion}")
    return predicted_emotion
    print("Raw prediction probabilities:", prediction)



#
# try:
#     text, audio_path = recording()
# except :
#     audio_path = "angry.mp3"
#     result_text = model.transcribe(audio_path, fp16=False)
#     text = result_text["text"]

audio_path = "C:\\Users\\acer\\PycharmProjects\\speech_emotion_mood\\angry.mp3"
result_text = sst_model.transcribe(audio_path, fp16=False)
text = result_text["text"]

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    )

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["sentence","prediction"],
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

# Create a LangChain LLMChain
emotion_chain = LLMChain(llm=llm, prompt=prompt)

# Example usage

result = emotion_chain.run(sentence = text , prediction=model_prediction(audio_path))

# Print the result
print("Input Sentence:", text)

print("Predicted Emotion:", result)






