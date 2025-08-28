
# backend/main.py
# main.py
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.hash import bcrypt
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime



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

# ---------------- EmpathyScore model ----------------
class EmpathyScore(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="scores")

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

# ---------------- Save score endpoint ----------------
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

# ---------------- Get user scores (Past scores) ----------------
@app.get("/user_scores/{username}")
def get_user_scores(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    scores = db.query(EmpathyScore).filter(EmpathyScore.user_id == user.id).order_by(EmpathyScore.timestamp.desc()).all()
    return [
        {"text": s.text, "score": s.score, "timestamp": s.timestamp.isoformat()}
        for s in scores
    ]

# ---------------- Show all users ----------------
@app.get("/show_users")
def show_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username} for u in users]

# ---------------- Show all users with hashed password (debug) ----------------
@app.get("/show_users_debug")
def show_users_debug(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username, "hashed_password": u.password} for u in users]

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