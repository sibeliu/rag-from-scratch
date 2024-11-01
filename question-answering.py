import os
import sys
import openai 
from openai import OpenAI
from os.path import join, dirname
from dotenv import load_dotenv
import fitz
import sqlite3
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Please set the environment variable OPENAI_API_KEY.")
    sys.exit(1)

# Read the question from the command line
if len(sys.argv) != 3:
    print("Usage: python question-answering.py /path/to/document 'question goes here'")
    sys.exit(1)

path = sys.argv[1]
question = sys.argv[2]

# Initialize the SentenceTransformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a SQLite database engine
# Define the database file path
db_path = 'text_chunks.db'

# Check if the database file exists and delete it if it does (for this demo, show end to end)
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Deleted existing database file: {db_path}")

# Create a new database
engine = create_engine(f'sqlite:///{db_path}', echo=True)
print(f"Created new database: {db_path}")

Base = declarative_base()

# Define the TextChunk to use with the database
class TextChunk(Base):
    __tablename__ = 'text_chunks'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    embedding = Column(LargeBinary)

# Create the table
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# helper functions
def load_pdf(file_path):
    """Load text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text):
    """Chunk the text using semantic chunking. Modify gradient threshold and min_chunk_size to improve results."""
    text_splitter = SemanticChunker(
                OpenAIEmbeddings(), 
                breakpoint_threshold_type="gradient",
                breakpoint_threshold_amount=.94,
                min_chunk_size=500
    )
    chunks = text_splitter.split_text(text)
    return chunks

def save_chunks_to_db(chunks):
    """Save text chunks and their embeddings to the database."""
    for chunk in chunks:
        embedding = model.encode([chunk])[0]
        embedding_bytes = embedding.tobytes()
        text_chunk = TextChunk(text=chunk, embedding=embedding_bytes)
        session.add(text_chunk)
    session.commit()

def hybrid_search(query, top_k=3): #implemented for broader context in future
    """Perform hybrid search over both vector embeddings and keywords."""
    # Generate the query embedding
    query_embedding = model.encode([query])[0]
    
    # Fetch all chunks from the database
    chunks = session.query(TextChunk).all()
    
    # Calculate cosine similarity for vector search
    similarities = []
    for chunk in chunks:
        chunk_embedding = np.frombuffer(chunk.embedding, dtype=np.float32)
        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        similarities.append((chunk, similarity))
    
    # Sort by similarity and by keywords
    similarities.sort(key=lambda x: x[1], reverse=True)
    keyword_matches = [chunk for chunk, _ in similarities if query.lower() in chunk.text.lower()]
    
    # Combine results
    hybrid_results = keyword_matches + [chunk for chunk, _ in similarities if chunk not in keyword_matches]
    
    return hybrid_results[:top_k]

def generate_response(base_prompt, user_question, context):
    user_prompt = f"Context:\n{context}\n\nUser Question: {user_question}"
    
    client = OpenAI()

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.2
    )
    return response.choices[0].message #response.choices[0].text.strip()

##################################
# Main logic
# Load data, chunk it

text = load_pdf(path)
chunks = chunk_text(text)

#save the chunks to database: we want to be able to perform hybrid search for both vector similarity and BM25
save_chunks_to_db(chunks)
results = hybrid_search(question) #returning a list of top-k relevant chunks

base_prompt = "You are a helpful assistant. Use the following context to answer the user's question."
context = results[0].text #we're only going to use the top chunk in this demo
context_id = results[0].id 

# Output to user
print ("\nYou asked:", question)
print ("\nThe relevant context found in the provided document is:\n   ", context)
print (f"\nThis context is found at ID {context_id} in the database.")

# Generate the response
response = generate_response(base_prompt, question, context)

print("\nGenerated Response using given context and gpt-4o:\n", response.content)