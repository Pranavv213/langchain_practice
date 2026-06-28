from langchain_ollama import ChatOllama
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the PDF documents
loader = OpenDataLoaderPDFLoader(
    file_path="sample.pdf",
    format="text"
)
documents = loader.load()

# 2. Split the documents
# Bumped chunk_size to 500 so sentences don't get chopped mid-thought!
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 4. Initialize and populate the in-memory vector store
vector_store = InMemoryVectorStore(embedding=embeddings)
vector_store.add_documents(chunks)

# 7. Initialize the ChatOllama model (Pulled out of the loop for efficiency)
# Make sure you have pulled this model locally via `ollama run llama3`
chat_model = ChatOllama(model="llama3.2:3b", temperature=0)

# Define a prompt template to guide the LLM on how to use the context
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using only the provided context below.\n\nContext:\n{context}"),
    ("human", "{query}")
])

while True:
    # 5. Get user input
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    
    # 6. Retrieve relevant documents from the vector store
    relevant_docs = vector_store.similarity_search(query, k=3)

    # 8. Format the prompt with the retrieved data
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    formatted_prompt = prompt_template.format_messages(context=context, query=query)

    # 9. Generate and print the response
    response = chat_model.invoke(formatted_prompt)
    print("\nResponse:", response.content)