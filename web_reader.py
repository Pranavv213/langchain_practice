from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import re

def extract_content(docs):
    """
    Takes a list of LangChain Document objects and returns
    a list of cleaned raw strings.
    """
    cleaned_docs = []
    for d in docs:
        text = d.page_content
        text = text.replace("\ue000", "")
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        cleaned_docs.append(text.strip())
    return cleaned_docs

FILE_PATH = "https://routerprotocol.com/"

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()

cleaned_texts = extract_content(docs)
print_text = "\n".join(cleaned_texts)

# Recommended: Bump up chunk_size so sentences don't get cut in half!
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(print_text)  # Generates a list of strings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embedding=embeddings)

# ─── THE FIX: Use add_texts instead of add_documents ───
vector_store.add_texts(chunks)

chat_model = ChatOllama(model="llama3.2:3b", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using only the provided context below.\n\nContext:\n{context}"),
    ("human", "{query}")
])

while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    
    relevant_docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    formatted_prompt = prompt_template.format_messages(context=context, query=query)

    response = chat_model.invoke(formatted_prompt)
    print("\nResponse:", response.content)