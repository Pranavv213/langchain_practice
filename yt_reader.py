from yt_dlp import YoutubeDL
import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import numpy as np

# 1. Configuration variables
url = "https://www.youtube.com/shorts/xZDMxh51kT8"
audio_filename = "audio"  # The base name for your audio file

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": f"{audio_filename}.%(ext)s",
    
    # Mac Fix: Explicitly tells yt-dlp where homebrew installs FFmpeg
    "ffmpeg_location": "/opt/homebrew/bin", 
    
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

# 2. Download and convert the YouTube audio
with YoutubeDL(ydl_opts) as ydl:
    print("Downloading and converting to MP3...")
    ydl.download([url])

print("Download complete!")

# 3. Load Whisper and transcribe the downloaded file
audio_file_path = f"{audio_filename}.mp3"

print("\nLoading Whisper model (this may take a moment on the first run)...")
model = whisper.load_model("base")

print(f"Transcribing {audio_file_path}...")
result = model.transcribe(audio_file_path)

# 4. Print and save the transcription
print("\n--- Transcription Result ---")
print(result["text"])
print("----------------------------")

# Added back the save to file operation to match your print statement below
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
print("Done! Text saved to transcription.txt")

# 5. Text Chunking (Optimized chunk size for character count)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(result["text"])

# 6. Embeddings and Vector Store
print("\nGenerating embeddings and indexing in vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_texts(chunks)

# 7. Initialize Ollama LLM
chat_model = ChatOllama(model="llama3.2:3b", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using only the provided context below.\n\nContext:\n{context}"),
    ("human", "{query}")
])

# 8. Interactive QA Loop
while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    
    # Retrieve top 3 relevant chunks
    relevant_docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    formatted_prompt = prompt_template.format_messages(context=context, query=query)

    # Invoke LLM
    response = chat_model.invoke(formatted_prompt)
    print("\nResponse:", response.content)
    