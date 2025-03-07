from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Interaction(Base):
    """Model for storing speech interactions with emotion analysis."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    transcript = Column(String, nullable=False)
    emotion_label = Column(String, nullable=False)
    confidence_score = Column(Float)
    user_id = Column(String, nullable=True)  # Optional for GDPR compliance
    audio_duration = Column(Float)  # Duration in seconds

def init_db(db_path: str) -> sessionmaker:
    """Initialize the database and return a session factory."""
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def store_interaction(
    session_factory: sessionmaker,
    transcript: str,
    emotion_label: str,
    confidence_score: Optional[float] = None,
    user_id: Optional[str] = None,
    audio_duration: Optional[float] = None,
) -> Interaction:
    """Store a new interaction in the database."""
    Session = session_factory
    session = Session()
    
    try:
        interaction = Interaction(
            transcript=transcript,
            emotion_label=emotion_label,
            confidence_score=confidence_score,
            user_id=user_id,
            audio_duration=audio_duration,
        )
        session.add(interaction)
        session.commit()
        return interaction
    finally:
        session.close()

def get_user_interactions(
    session_factory: sessionmaker, user_id: str
) -> list[Interaction]:
    """Retrieve all interactions for a specific user."""
    Session = session_factory
    session = Session()
    
    try:
        return session.query(Interaction).filter_by(user_id=user_id).all()
    finally:
        session.close()

def delete_user_data(session_factory: sessionmaker, user_id: str) -> None:
    """Delete all data associated with a user (GDPR compliance)."""
    Session = session_factory
    session = Session()
    
    try:
        session.query(Interaction).filter_by(user_id=user_id).delete()
        session.commit()
    finally:
        session.close() 