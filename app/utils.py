import os, hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration constants
CHROMA_PATH = "chroma"
DATA_PATH   = "data/books"
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_SCORE   = 0.1

def get_embedding_db() -> Chroma:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

def calculate_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# Synchronous email function (no extra packages needed)
def send_notification_email(question: str, user_info: str = "Unknown"):
    """Send email notification when question cannot be answered"""
    
    # Email configuration from .env file
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    NOTIFICATION_EMAIL = "help.dwexo@gmail.com"
    
    if not EMAIL_USER or not EMAIL_PASSWORD:
        print("Email credentials not configured")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = NOTIFICATION_EMAIL
        msg['Subject'] = "DWEXO Assistant - Unanswered Question Alert"
        
        body = f"""
DWEXO Assistant Alert

A user asked a question that could not be answered from the available documents.

Question: {question}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please review and consider adding relevant documentation to improve the knowledge base.

---
DWEXO Assistant System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_USER, NOTIFICATION_EMAIL, text)
        server.quit()
        
        print(f"Notification email sent for question: {question}")
        return True
        
    except Exception as e:
        print(f"Failed to send notification email: {e}")
        return False