# aii.py

import streamlit as st
import requests
import speech_recognition as sr
from gtts import gTTS
import os
import uuid
from googletrans import Translator
from transformers import pipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------- Translator -----------------
translator = Translator()

# ----------------- Streamlit UI Config -----------------
st.set_page_config(page_title="🌾 AI Helper for Farmers", page_icon="🌾", layout="wide")

# ----------------- Voice Output -----------------
def speak(text, language_code='hi'):
    if not text.strip():
        return
    try:
        filename = f"voice_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang=language_code)
        tts.save(filename)
        audio = open(filename, 'rb').read()
        st.audio(audio, format='audio/mp3')
        os.remove(filename)
    except Exception as e:
        st.error(f"🎧 Voice error: {e}")

# ----------------- Voice Input -----------------
def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        st.warning("❌ Could not recognize your voice.")
        return ""

# ----------------- Crop Recommendation -----------------
def recommend_crop(soil_type, season):
    rules = {
        ("Loamy", "Kharif"): "Rice",
        ("Sandy", "Rabi"): "Wheat",
        ("Clay", "Zaid"): "Maize",
        ("Loamy", "Rabi"): "Barley",
        ("Black Soil", None): "Cotton",
        ("Red Soil", None): "Groundnut",
        ("Alluvial Soil", None): "Sugarcane",
        ("Laterite Soil", None): "Tea",
        ("Peaty Soil", None): "Paddy"
    }
    return rules.get((soil_type, season), rules.get((soil_type, None), "Try mixed cropping or consult local experts."))

# ----------------- Fertilizer Suggestion -----------------
fertilizers = {
    "Rice": "Urea, DAP, Potash (मिट्टी में नाइट्रोजन, फॉस्फोरस और पोटाश बढ़ाता है)",
    "Wheat": "Urea, SSP, Potash (नाइट्रोजन, फॉस्फोरस और पोटाश प्रदान करता है)",
    "Maize": "Urea, DAP, Zinc Sulphate (नाइट्रोजन, फॉस्फोरस और जिंक के लिए)",
    "Barley": "Urea, Ammonium Sulphate (नाइट्रोजन बढ़ाता है)"
}

def suggest_fertilizer(crop):
    return fertilizers.get(crop, "Try organic or local fertilizers.")

# ----------------- Weather API -----------------
def get_weather(city):
    api_key = "a6f81aff8e354cf14db2c448cbb27e5c"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=100)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error("❌ Failed to fetch weather data. Please check your internet connection.")
        return {}



# ----------------- AI Chat (Offline) -----------------
def ai_farmer_chat_offline(prompt):
    prompt = prompt.lower()
    if "rice" in prompt: return "Use resistant seeds, proper water management, and timely fungicide."
    if "wheat" in prompt: return "Ensure proper fertilization and pest control for wheat."
    return "Ask about crops, fertilizers, or disease control."

# ----------------- AI Chat (Local Model) -----------------
@st.cache_resource
def load_local_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

local_model = load_local_model()

def ai_farmer_chat_local(prompt):
    final_prompt = f"You are an expert farmer. Question: {prompt}"
    response = local_model(
        final_prompt,
        max_length=150,
        temperature=0.7,
        repetition_penalty=1.5,
        num_return_sequences=1
    )[0]['generated_text']
    return response.strip()

# ----------------- Crop Detection Model -----------------
@st.cache_resource
def load_plant_model():
    return load_model("plant_model.h5")  # ✅ Your final trained model

model = load_plant_model()
labels = open("labels.txt").read().splitlines()

def predict_crop(file):
    img = image.load_img(file, target_size=(224, 224))
    img_arr = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
    pred = model.predict(img_arr)
    index = np.argmax(pred)
    return labels[index].strip(), float(np.max(pred))

# ----------------- Fill-Mask Model -----------------
@st.cache_resource
def load_fill_mask_model():
   return pipeline('text2text-generation', model='google/flan-t5-small',framework='pt')


local_model = load_local_model()

# ----------------- UI -----------------
st.markdown("""
    <style>
        .main-title {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white; padding: 20px; border-radius: 12px; text-align: center;
        }
        .feature-box {
            background-color: #e6f2ff; padding: 20px; border-radius: 12px;
            text-align: center;
        }
        .stButton>button {
            background-color: #00b894; color: white; border-radius: 8px; padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-title">
        <h1>🌾 AI Helper for Farmers</h1>
        <h4>Your All-in-One Smart Farming Partner</h4>
    </div>
""", unsafe_allow_html=True)

menu = ["Home", "Crop Recommendation", "Weather Info", "Fertilizer Suggestion",
        "AI Farmer Chat (Offline)", "AI Farmer Chat (Local Model)", 
        "Plant Crop Detection"]
choice = st.sidebar.selectbox("Menu", menu)

# ----------------- Pages -----------------
if choice == "Home":
    st.image("https://cdn.pixabay.com/photo/2017/06/19/11/02/agriculture-2416350_1280.jpg", use_container_width=True)
    st.markdown("### 🌟 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="feature-box" style="margin-bottom: 20px;">
                <h4>🌱 Crop Recommendation</h4>
                <p>Get the best crops based on soil type and season.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-box" style="margin-bottom: 20px;">
                <h4>🌦️ Weather Info</h4>
                <p>Live weather updates for your city.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-box" style="margin-bottom: 20px;">
                <h4>💧 Fertilizer Suggestion</h4>
                <p>Get the right fertilizers for better crop yield.</p>
            </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
            <div class="feature-box" style="margin-bottom: 20px;">
                <h4>🤖 AI Chat (Offline)</h4>
                <p>Quick offline farming advice anytime.</p>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
            <div class="feature-box" style="margin-bottom: 20px;">
                <h4>🔠️ AI Chat (Voice/Text)</h4>
                <p>Ask farming questions in multiple languages.</p>
            </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
            <div class="feature-box" style="margin-bottom: 20px;">
                <h4>🌿 Disease Detection</h4>
                <p>Upload plant images to detect diseases using AI.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <hr>
        <div style="text-align: center;">
            <h4>🚀 Let's make farming Smarter, Faster, and Easier together! 🌾</h4>
        </div>
    """, unsafe_allow_html=True)

elif choice == "Crop Recommendation":
    st.subheader("🌱 Recommend Crop")
    soil = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay", "Black Soil", "Red Soil", "Alluvial Soil", "Laterite Soil", "Peaty Soil"])
    season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
    if st.button("Recommend"):
        crop = recommend_crop(soil, season)
        st.success(f"Recommended Crop: {crop}")

elif choice == "Weather Info":
    st.subheader("🌦️ Current Weather Information")
    city = st.text_input("Enter your City Name")

    if st.button("Get Weather"):
        data = get_weather(city)

        if data.get("cod") == 200:
            st.subheader(f"Weather in {data['name']}, {data['sys']['country']}")
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            weather_desc = data['weather'][0]['description'].capitalize()
            wind_speed = data['wind']['speed']
            icon_url = f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"

            st.write(f"🌡️ Temperature: {temperature} °C")
            st.write(f"💧 Humidity: {humidity}%")
            st.write(f"🌥️ Weather: {weather_desc}")
            st.write(f"💨 Wind Speed: {wind_speed} m/s")
            st.image(icon_url, width=100)

            # Crop Suggestion Logic
            def suggest_crops(temp, weather):
                weather = weather.lower()
                if 'rain' in weather or 'shower' in weather:
                    return ['Paddy', 'Maize', 'Sugarcane', 'Groundnut']
                elif 'clear' in weather or 'sun' in weather:
                    if temp > 30:
                        return ['Cotton', 'Bajra', 'Millets', 'Soybean']
                    else:
                        return ['Wheat', 'Barley', 'Mustard', 'Peas']
                elif 'cloud' in weather:
                    return ['Rice', 'Pulses', 'Vegetables']
                elif 'snow' in weather:
                    return ['No major cropping possible - Extreme cold']
                else:
                    return ['General crops like vegetables, pulses, and grains']

            suggested_crops = suggest_crops(temperature, weather_desc)

            st.subheader("🌾 Suitable Crops for Current Weather:")
            for crop in suggested_crops:
                st.write(f"✅ {crop}")
        else:
            st.error("City not found. Please enter a valid city name.")

elif choice == "Fertilizer Suggestion":
    st.subheader("💧 Suggest Fertilizer")
    crop = st.text_input("Enter Crop Name")
    if st.button("Suggest"):
        result = suggest_fertilizer(crop.title())
        st.success(f"Suggested: {result}")
        speak(result)
elif choice == "AI Farmer Chat (Offline)":
    st.subheader("🤖 Chat with AI Farmer Assistant (Offline Mode)")
    user_input = st.text_area("Ask your farming question")
    if st.button("Get Answer (Offline)"):
        if user_input.strip():
            answer = ai_farmer_chat_offline(user_input)
            st.write(f"**AI Farmer:** {answer}")
        else:
            st.warning("Please enter a question.")

if choice == "AI Farmer Chat (Local Model)":
    st.subheader("🤖 Chat with AI Farmer Assistant (Auto Language Detection)")

    st.subheader("💬 Text Input")
    user_input = st.text_area("Ask your farming question in any language")

    if st.button("Get Answer"):
        if user_input.strip():
            with st.spinner('🤖 Thinking...'):
                detected_lang = translator.detect(user_input).lang
                answer_en = ai_farmer_chat_local(user_input)
                if detected_lang != 'en':
                    final_answer = translator.translate(answer_en, src='en', dest=detected_lang).text
                else:
                    final_answer = answer_en

            st.success(f"**AI Farmer Answer:** {final_answer}")
            speak(final_answer, language_code=detected_lang)
        else:
            st.warning("⚠️ Please enter a question.")

    if st.button("Clear Chat"):
        st.rerun()

    st.subheader("🎤 Voice Input")
    if st.button("Ask by Voice"):
        question = voice_input()
        if question:
            with st.spinner('🤖 Thinking...'):
                detected_lang = translator.detect(question).lang
                answer_en = ai_farmer_chat_local(question)
                if detected_lang != 'en':
                    final_answer = translator.translate(answer_en, src='en', dest=detected_lang).text
                else:
                    final_answer = answer_en

            st.success(f"**You asked:** {question}")
            st.success(f"**AI Farmer Answer:** {final_answer}")
            speak(final_answer, language_code=detected_lang)


elif choice == "Plant Crop Detection":
    st.subheader("🌿 Detect Crop from Leaf")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=500)
        if st.button("Detect"):
            name, conf = predict_crop(uploaded_file)
            st.success(f"Crop: {name}")
            st.info(f"Confidence: {conf*100:.2f}%")
            plant_disease_info = {
                "jute": "Needs warm, humid climate and loamy soil.",
                "rice": "Flooded fields. Keep weeds away.",
                "wheat": "Avoid overwatering. Monitor for rust.",
                "sugarcane": "Needs heavy irrigation and compost.",
                "maize": "Requires nitrogen-rich soil and sunlight."
            }
            recommendation = plant_disease_info.get(name.lower(), "No info available.")
            st.markdown(f"**Recommendation:** {recommendation}")
            speak(f"{name} detected. {recommendation}", language_code='hi')

