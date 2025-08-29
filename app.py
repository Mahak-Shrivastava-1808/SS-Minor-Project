# frontend/app.py
# app.py (Streamlit frontend)
import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import plotly.graph_objects as go
import streamlit.components.v1 as components
# groq optional import
try:
    from groq import Groq
except Exception:
    Groq = None
from dotenv import load_dotenv
import os
import requests

# -------------------- Load OpenAI / GROQ API Key --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if (Groq is not None and GROQ_API_KEY) else None

# Backend URL (change if your backend runs elsewhere)
API_URL = os.getenv("EMPATHY_API_URL", "http://127.0.0.1:8000")

# -------------------- Browser-based Speak --------------------
def speak(text: str):
    if text and isinstance(text, str):
        components.html(
            f"""
            <script>
                var msg = new SpeechSynthesisUtterance("{text}");
                msg.lang = "en-US";
                window.speechSynthesis.speak(msg);
            </script>
            """,
            height=0,
        )

# -------------------- Sentiment Analysis --------------------
def analyze_emotion(text: str):
    blob = TextBlob(text)
    score = float(blob.sentiment.polarity)
    if score > 0.1:
        label_tb, icon = "Positive", "🙂"
    elif score < -0.1:
        label_tb, icon = "Negative", "😟"
    else:
        label_tb, icon = "Neutral", "😐"

    label_ai = "GROQ not configured or API key missing."
    if client:
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an advanced emotion analysis assistant. "
                            "Your job is to analyze user text and identify the underlying emotions. "
                            "Return output in this format:\n"
                            "Primary Emotion: <main emotion>\n"
                            "Secondary Emotions: <other possible emotions>\n"
                            "Confidence: <percentage>\n"
                            "Explanation: <short reasoning>"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the emotions in this text: {text}",
                    },
                ],
                max_tokens=120,
            )
            label_ai = response.choices[0].message.content.strip()
        except Exception as e:
            label_ai = f"Error: {e}"

    return label_tb, icon, score, label_ai

# -------------------- New Feature 1: Email Analysis --------------------
def analyze_email(email_text: str):
    if client:
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant that analyzes emails. "
                        "Detect the tone (formal/informal/neutral), "
                        "politeness level (0-100), and emotional intent. "
                        "Return output in this format:\n"
                        "Tone: <tone>\n"
                        "Politeness: <score>/100\n"
                        "Emotional Intent: <intent>"
                    )},
                    {"role": "user", "content": f"Analyze this email: {email_text}"}
                ],
                max_tokens=120,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"
    return "API key not set."

# -------------------- Gauge Meter --------------------
def show_gauge(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Sentiment Score"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [-1, -0.3], "color": "red"},
                    {"range": [-0.3, 0.3], "color": "lightgray"},
                    {"range": [0.3, 1], "color": "green"},
                ],
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Speech Recognition --------------------
def recognize_speech():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Adjusting for background noise…")
            r.adjust_for_ambient_noise(source, duration=1)
            st.info("🎤 Listening… Speak now.")
            audio = r.listen(source, timeout=5, phrase_time_limit=8)
        return r.recognize_google(audio)
    except Exception:
        return None

# -------------------- Page Config + CSS --------------------
st.set_page_config(page_title="Empathy Meter", page_icon="🎭", layout="centered")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Bubblegum+Sans&display=swap" rel="stylesheet">

<style>
/* Background */
.stApp {
  background: linear-gradient(to right, #093637, #44A08D);
  color: white;
  font-family: 'Inter', sans-serif;
}

/* Center & lock the LOGIN page to viewport */
body.page-login main .block-container{
  min-height: 100vh;                 /* fill full screen */
  display: flex;
  flex-direction: column;
  justify-content: center;           /* vertical center */
  align-items: center;               /* horizontal center */
  max-width: 520px;                  /* nice, compact card width */
  padding: 0 !important;
}
body.page-login { overflow: hidden; } /* no scroll on login */

/* Global text fix */
.stTextInput label,
.stPasswordInput label,
.stTextArea label,
div, p, span, label {
  color: white !important;
}

/* Inputs */
.stTextInput input,
.stPasswordInput input,
.stTextArea textarea {
  background: #093637 !important;
  border: 2px solid rgba(0, 200, 255, 0.25) !important;
  border-radius: 12px !important;
  color: white !important;
  padding: 10px !important;
  font-size: 1rem !important;
  transition: 0.3s ease-in-out;
}
.stTextInput input:focus,
.stPasswordInput input:focus,
.stTextArea textarea:focus {
  border: 2px solid #00c8ff !important;
  box-shadow: 0 0 10px #00c8ff !important;
  outline: none !important;
}

/* Buttons: side by side */
.stButton > button {
  width: 100%;
  background: linear-gradient(135deg, #093637, #44A08D) !important;
  border: 2px solid rgba(0, 200, 255, 0.25) !important;
  border-radius: 12px !important;
  color: white !important;
  font-weight: bold !important;
  padding: 10px 20px !important;
  transition: all 0.3s ease-in-out;
  box-shadow: 0 0 10px rgba(0, 200, 255, 0.25);
}
.stButton > button:hover {
  background: linear-gradient(135deg, #44A08D, #093637) !important;
  transform: scale(1.02);
  box-shadow: 0 0 20px rgba(0, 200, 255, 0.45);
}

/* Glass Card */
.glass {
  background: rgba(255,255,255,0.08);
  border: 2px solid rgba(0, 200, 255, 0.25);
  border-radius: 20px;
  box-shadow: 0 0 20px rgba(0, 200, 255, 0.25), 0 8px 40px rgba(0,0,0,0.4);
  backdrop-filter: blur(14px);
  padding: 20px;
  font-family: 'Bubblegum Sans', cursive;
  font-size: 2rem;
  animation: pulseGlow 2s infinite;
  text-align:center;
  margin: auto;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass:hover {
  transform: translateY(-6px);
  box-shadow: 0 0 30px rgba(0, 200, 255, 0.45), 0 12px 50px rgba(0,0,0,0.6);
}

/* Floating Emojis */
@keyframes floatY {
  0% { transform: translateY(0); opacity: 0.9; }
  50% { transform: translateY(-12px); opacity: 1; }
  100% { transform: translateY(0); opacity: 0.9; }
}
.floater { position: fixed; font-size: 35px; opacity: .7; animation: floatY 8s infinite; }
.f1 { top: 8%; left: 8%; animation-delay: 0s;  }
.f2 { top: 30%; right: 10%; animation-delay: .6s; }
.f3 { bottom: 10%; left: 12%; animation-delay: 1.2s; }
.f4 { bottom: 18%; right: 12%; animation-delay: 1.8s; }
.f5 { bottom: 15%; left: 50%; animation-delay: 2.2s; }

/* Glow animation */
@keyframes pulseGlow {
  0% { text-shadow: 0 0 6px #00c8ff; }
  50% { text-shadow: 0 0 18px #00c8ff; }
  100% { text-shadow: 0 0 6px #00c8ff; }
}
.title {
  font-size: 2.2rem; font-weight: 800;
  text-align: center;
  animation: pulseGlow 2s infinite;
}
.subtitle {
  font-size: 1rem; text-align: center;
  opacity: .95; margin-bottom: 18px;
}
</style>
<div class="floater f1">✨</div>
<div class="floater f2">🫡</div>
<div class="floater f3">🥶</div>
<div class="floater f4">🫣</div>
<div class="floater f5">🫢</div>
""",
    unsafe_allow_html=True,
)

# -------------------- Session --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "login"

# -------------------- Pages --------------------
def login_page():
    components.html("<script>document.body.classList.add('page-login');</script>", height=0)

    st.markdown('<div class="glass">🙂‍↔ Welcome to Empathy Meter</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">🔐 Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sign in to access Empathy Meter</div>', unsafe_allow_html=True)

    user = st.text_input("Username")
    pwd  = st.text_input("Password", type="password")

    msg_placeholder = st.empty()
    msg_placeholder.markdown("<div style='height:44px'></div>", unsafe_allow_html=True)

    # --- Buttons side by side ---
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Login", key="login_btn", use_container_width=True):
            try:
                res = requests.post(f"{API_URL}/login", json={"username": user, "password": pwd}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.logged_in = True
                    st.session_state.username = data.get("username")
                    # store token if present (frontend code expects it in some earlier version)
                    st.session_state.token = data.get("access_token")
                    msg_placeholder.success("✅ Login successful!")
                    speak("Login successful. Welcome to Empathy Meter.")
                    st.rerun()
                else:
                    msg_placeholder.error(res.json().get("detail", "Invalid credentials"))
                    speak("Invalid username or password.")
            except Exception as e:
                msg_placeholder.error(f"⚠ Backend error: {e}")

    with col2:
        if st.button("🆕 Sign Up", key="signup_btn", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()

def signup_page():
    components.html("<script>document.body.classList.remove('page-login');</script>", height=0)

    st.markdown('<div class="glass">📝 Create Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">✨ Sign Up</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Register to use Empathy Meter</div>', unsafe_allow_html=True)

    new_user = st.text_input("Choose Username")
    new_pwd = st.text_input("Choose Password", type="password")
    confirm_pwd = st.text_input("Confirm Password", type="password")

    if st.button("✅ Register"):
        if not new_user or not new_pwd:
            st.warning("⚠ All fields are required.")
            speak("All fields are required.")
        elif new_pwd != confirm_pwd:
            st.error("⚠ Passwords do not match.")
            speak("Passwords do not match.")
        else:
            try:
                res = requests.post(f"{API_URL}/signup", json={"username": new_user, "password": new_pwd}, timeout=10)
                if res.status_code == 200:
                    st.success("🎉 Account created successfully! Please login.")
                    speak("Account created successfully. Please login now.")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(res.json().get("detail", "Signup failed"))
            except Exception as e:
                st.error(f"⚠ Backend error: {e}")

    if st.button("🔙 Back to Login"):
        st.session_state.page = "login"
        st.rerun()

def empathy_page():
    components.html("<script>document.body.classList.remove('page-login');</script>", height=0)

    st.markdown('<div class="glass">🎭 Empathy Meter</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Type or speak — I will analyze and speak out the result with score.</div>', unsafe_allow_html=True)

    st.caption(f"Logged in as: *{st.session_state.username}*")

    st.subheader("✍ Text Input")
    text_in = st.text_area("Type how you feel…", height=120)
    if st.button("🔍 Analyze Text"):
        if text_in.strip():
            st.success(f"📝 You wrote: {text_in}")
            speak(f"You wrote: {text_in}")
            label_tb, icon, score, label_ai = analyze_emotion(text_in)
            st.info(f"{icon} Emotion Detector: {label_tb}")
            st.info(f"🤖 Astor: {label_ai}")
            speak(f"This sounds {label_tb}.")
            show_gauge(score)

            # Save to backend
            try:
                requests.post(f"{API_URL}/submit_score", json={
                    "username": st.session_state.username,
                    "text": text_in,
                    "score": score
                }, timeout=10)
            except Exception as e:
                st.error(f"⚠ Could not save score: {e}")
        else:
            st.warning("Please enter some text.")
            speak("Please enter text first.")

    st.divider()

    st.subheader("🎤 Voice Input")
    if st.button("👂 Start Recording"):
        with st.spinner("Listening…"):
            heard = recognize_speech()
        if not heard:
            st.error("Could not understand speech.")
            speak("I could not understand you. Please try again.")
        else:
            st.success(f"🗣 You said: {heard}")
            speak(f"You said: {heard}")
            label_tb, icon, score, label_ai = analyze_emotion(heard)
            st.info(f"{icon} TextBlob: {label_tb}")
            st.info(f"🤖 Astor: {label_ai}")
            speak(f"This sounds {label_tb}.")
            show_gauge(score)

            # Save to backend
            try:
                requests.post(f"{API_URL}/submit_score", json={
                    "username": st.session_state.username,
                    "text": heard,
                    "score": score
                }, timeout=10)
            except Exception as e:
                st.error(f"⚠ Could not save score: {e}")

    st.divider()

    # ---- New Feature: Email ----
    st.subheader("📧 Email Tone Analysis")
    email_text = st.text_area("Paste your email here…", height=150, key="email_text")
    if st.button("🔍 Analyze Email"):
        if email_text.strip():
            result = analyze_email(email_text)
            st.info(result)
            speak(result)
        else:
            st.warning("Please paste an email first.")

    st.divider()

    # Optional: Score History viewer (keeps UI as-is)
    with st.expander("📊 View Score History"):
        if st.button("Refresh History"):
            st.experimental_rerun()
        try:
            res = requests.get(f"{API_URL}/user_scores/{st.session_state.username}", timeout=10)
            if res.status_code == 200:
                scores = res.json()
                if not scores:
                    st.caption("No scores yet.")
                else:
                    for s in scores:
                        st.write(f"📝 *{s['text']}* → *{s['score']:.3f}*  (at {s['timestamp']})")
                    # Plot line chart of scores
                    values = [item["score"] for item in reversed(scores)]
                    fig = go.Figure([go.Scatter(y=values, mode="lines+markers")])
                    fig.update_layout(title="Empathy Score History (most recent on right)", yaxis_title="Score (-1 to 1)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(res.json().get("detail", "Could not fetch scores"))
        except Exception as e:
            st.error(f"⚠ Could not fetch scores: {e}")

    if st.button("🔙 Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.session_state.username = None
        st.session_state.token = None
        st.rerun()

# -------------------- Router --------------------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
else:
    empathy_page()