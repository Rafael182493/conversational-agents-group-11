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
    db_session = session_factory()
    try:
        new_session = Session(
            start_time=datetime.utcnow(),
            user_id=user_id
        )
        db_session.add(new_session)
        db_session.commit()
        
        # Refresh the session to get the generated ID
        db_session.refresh(new_session)
        
        # Create a copy of the relevant attributes
        session_id = new_session.id
        session_start = new_session.start_time
        session_user = new_session.user_id
        
        # Create a new detached session object with the copied attributes
        return Session(
            id=session_id,
            start_time=session_start,
            user_id=session_user
        )
    finally:
        db_session.close()

def end_session(session_factory: sessionmaker, session_id: int) -> None:
    """End a session by setting its end time."""
    db_session = session_factory()
    try:
        session = db_session.query(Session).filter_by(id=session_id).first()
        if session:
            session.end_time = datetime.utcnow()
            db_session.commit()
    finally:
        db_session.close()

def store_interaction(
    session_factory: sessionmaker,
    session_id: int,
    transcript: str,
    emotion_label: str,
    confidence_score: Optional[float] = None,
    audio_duration: Optional[float] = None,
) -> Interaction:
    """Store a new interaction in the database."""
    db_session = session_factory()
    
    try:
        interaction = Interaction(
            session_id=session_id,
            transcript=transcript,
            emotion_label=emotion_label,
            confidence_score=confidence_score,
            audio_duration=audio_duration,
        )
        db_session.add(interaction)
        db_session.commit()
        return interaction
    finally:
        db_session.close()

def get_session_interactions(
    session_factory: sessionmaker, session_id: int
) -> List[Interaction]:
    """Retrieve all interactions for a specific session."""
    db_session = session_factory()
    
    try:
        return db_session.query(Interaction).filter_by(session_id=session_id).all()
    finally:
        db_session.close()

def get_user_sessions(
    session_factory: sessionmaker, user_id: str
) -> List[Session]:
    """Retrieve all sessions for a specific user."""
    db_session = session_factory()
    
    try:
        return db_session.query(Session).filter_by(user_id=user_id).all()
    finally:
        db_session.close()

def delete_user_data(session_factory: sessionmaker, user_id: str) -> None:
    """Delete all data associated with a user (GDPR compliance)."""
    db_session = session_factory()
    
    try:
        # This will cascade delete all associated interactions
        db_session.query(Session).filter_by(user_id=user_id).delete()
        db_session.commit()
    finally:
        db_session.close() 