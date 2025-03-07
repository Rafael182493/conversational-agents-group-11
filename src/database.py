from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class Session(Base):
    """Model for storing conversation sessions."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime)
    user_id = Column(String, nullable=True)  # Optional for GDPR compliance
    
    # Relationship to interactions
    interactions = relationship("Interaction", back_populates="session", cascade="all, delete-orphan")

class Interaction(Base):
    """Model for storing speech interactions with emotion analysis."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    transcript = Column(String, nullable=False)
    emotion_label = Column(String, nullable=False)
    confidence_score = Column(Float)
    audio_duration = Column(Float)  # Duration in seconds
    
    # Relationship to session
    session = relationship("Session", back_populates="interactions")

def init_db(db_path: str) -> sessionmaker:
    """Initialize the database and return a session factory."""
    engine = create_engine(f"sqlite:///{db_path}", echo=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def create_session(session_factory: sessionmaker, user_id: Optional[str] = None) -> Session:
    """Create a new session."""
    Session = session_factory
    session = Session()
    try:
        new_session = Session(user_id=user_id)
        session.add(new_session)
        session.commit()
        return new_session
    finally:
        session.close()

def end_session(session_factory: sessionmaker, session_id: int) -> None:
    """End a session by setting its end time."""
    Session = session_factory
    session = Session()
    try:
        db_session = session.query(Session).filter_by(id=session_id).first()
        if db_session:
            db_session.end_time = datetime.utcnow()
            session.commit()
    finally:
        session.close()

def store_interaction(
    session_factory: sessionmaker,
    session_id: int,
    transcript: str,
    emotion_label: str,
    confidence_score: Optional[float] = None,
    audio_duration: Optional[float] = None,
) -> Interaction:
    """Store a new interaction in the database."""
    Session = session_factory
    session = Session()
    
    try:
        interaction = Interaction(
            session_id=session_id,
            transcript=transcript,
            emotion_label=emotion_label,
            confidence_score=confidence_score,
            audio_duration=audio_duration,
        )
        session.add(interaction)
        session.commit()
        return interaction
    finally:
        session.close()

def get_session_interactions(
    session_factory: sessionmaker, session_id: int
) -> List[Interaction]:
    """Retrieve all interactions for a specific session."""
    Session = session_factory
    session = Session()
    
    try:
        return session.query(Interaction).filter_by(session_id=session_id).all()
    finally:
        session.close()

def get_user_sessions(
    session_factory: sessionmaker, user_id: str
) -> List[Session]:
    """Retrieve all sessions for a specific user."""
    Session = session_factory
    session = Session()
    
    try:
        return session.query(Session).filter_by(user_id=user_id).all()
    finally:
        session.close()

def delete_user_data(session_factory: sessionmaker, user_id: str) -> None:
    """Delete all data associated with a user (GDPR compliance)."""
    Session = session_factory
    session = Session()
    
    try:
        # This will cascade delete all associated interactions
        session.query(Session).filter_by(user_id=user_id).delete()
        session.commit()
    finally:
        session.close() 