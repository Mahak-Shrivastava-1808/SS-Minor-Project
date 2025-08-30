from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.hash import bcrypt
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import re

# ---------------- Database setup ----------------
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------- User model ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    scores = relationship("EmpathyScore", back_populates="user")
    emails = relationship("EmailAnalysis", back_populates="user")

# ---------------- EmpathyScore model ----------------
class EmpathyScore(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="scores")

# ---------------- EmailAnalysis model ----------------
class EmailAnalysis(Base):
    __tablename__ ="emails"
    id = Column(Integer, primary_key=True, index=True)
    email_text = Column(Text)
    tone = Column(String)
    politeness = Column(Integer)
    intent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="emails")

# Create tables
Base.metadata.create_all(bind=engine)

# ---------------- Pydantic schemas ----------------
class UserCreate(BaseModel):
    username: str
    password: str

class ScoreCreate(BaseModel):
    username: str
    text: str
    score: float

# Existing structured schema (if frontend sends parsed pieces)
class EmailCreate(BaseModel):
    username: str
    email_text: str
    tone: str
    politeness: int
    intent: str

# Flexible schema (also accepts a single combined 'analysis' string)
class EmailFlexible(BaseModel):
    username: str
    email_text: str
    # Either send these three...
    tone: str | None = None
    politeness: int | None = None
    intent: str | None = None
    # ...or send one combined analysis line like:
    # "Tone: Informal Politeness: 60/100 Emotional Intent: Impatience"
    analysis: str | None = None

# ---------------- App init ----------------
app = FastAPI(title="Empathy Meter Backend")

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Dependency ----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Signup endpoint ----------------
@app.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_pw = bcrypt.hash(user.password)
    new_user = User(username=user.username, password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User created successfully", "username": new_user.username}

# ---------------- Login endpoint ----------------
@app.post("/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not bcrypt.verify(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {
        "msg": "Login successful",
        "username": db_user.username,
        "access_token": "dummy_token_for_demo"
    }

# ---------------- Save Empathy Score ----------------
@app.post("/submit_score")
def submit_score(score: ScoreCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == score.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_score = EmpathyScore(text=score.text, score=score.score, user_id=user.id)
    db.add(new_score)
    db.commit()
    db.refresh(new_score)
    return {"msg": "Score saved successfully", "id": new_score.id}

# ---------------- Save Email Analysis (flexible) ----------------
# NOTE: Frontend agar single string bheje to bhi chalega, aur
# tone/politeness/intent alag-alag bheje to bhi.
@app.post("/submit_email_analysis")
def submit_email_analysis(payload: EmailFlexible, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    tone = payload.tone
    intent = payload.intent
    politeness = payload.politeness

    # If combined 'analysis' string is provided, parse it
    if payload.analysis:
        # Examples the frontend shows:
        # "Tone: Informal Politeness: 60/100 Emotional Intent: Impatience"
        tone_match = re.search(r"Tone:\s*([A-Za-z ]+)", payload.analysis)
        pol_match = re.search(r"Politeness:\s*(\d+)\s*/\s*100", payload.analysis)
        intent_match = re.search(r"(?:Emotional Intent|Intent):\s*([A-Za-z ]+)", payload.analysis)

        if tone is None and tone_match:
            tone = tone_match.group(1).strip()
        if politeness is None and pol_match:
            politeness = int(pol_match.group(1))
        if intent is None and intent_match:
            intent = intent_match.group(1).strip()

    # Final validation
    if tone is None or politeness is None or intent is None:
        raise HTTPException(
            status_code=400,
            detail="tone, politeness, and intent are required (either individually or parsable from 'analysis').",
        )

    rec = EmailAnalysis(
        user_id=user.id,
        email_text=payload.email_text,
        tone=tone,
        politeness=politeness,
        intent=intent,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return {
        "msg": "Email analysis stored successfully",
        "id": rec.id,
        "tone": rec.tone,
        "politeness": rec.politeness,
        "intent": rec.intent,
        "timestamp": rec.timestamp.isoformat(),
    }

# ---------------- Get user scores (Past scores) ----------------
@app.get("/user_scores/{username}")
def get_user_scores(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    scores = (
        db.query(EmpathyScore)
        .filter(EmpathyScore.user_id == user.id)
        .order_by(EmpathyScore.timestamp.desc())
        .all()
    )
    return [
        {"text": s.text, "score": s.score, "timestamp": s.timestamp.isoformat()}
        for s in scores
    ]

# ---------------- Get user emails (Past email analysis) ----------------
@app.get("/user_emails/{username}")
def get_user_emails(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    emails = (
        db.query(EmailAnalysis)
        .filter(EmailAnalysis.user_id == user.id)
        .order_by(EmailAnalysis.timestamp.desc())
        .all()
    )
    return [
        {
            "email_text": e.email_text,
            "tone": e.tone,
            "politeness": e.politeness,
            "intent": e.intent,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in emails
    ]

# ---------------- Show all users ----------------
@app.get("/show_users")
def show_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username} for u in users]

# ---------------- Show all past scores (all users) ----------------
@app.get("/all_scores")
def all_scores(db: Session = Depends(get_db)):
    scores = db.query(EmpathyScore).order_by(EmpathyScore.timestamp.desc()).all()
    return [
        {
            "id": s.id,
            "username": s.user.username if s.user else None,
            "text": s.text,
            "score": s.score,
            "timestamp": s.timestamp.isoformat(),
        }
        for s in scores
    ]

# ---------------- Show all past email analysis (all users) ----------------
@app.get("/all_emails")
def all_emails(db: Session = Depends(get_db)):
    emails = db.query(EmailAnalysis).order_by(EmailAnalysis.timestamp.desc()).all()
    return [
        {
            "id": e.id,
            "username": e.user.username if e.user else None,
            "email_text": e.email_text,
            "tone": e.tone,
            "politeness": e.politeness,
            "intent": e.intent,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in emails
    ]