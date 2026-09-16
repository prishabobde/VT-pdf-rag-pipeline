from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()


pdf_files = [
    "data/report1.pdf",
    "data/report2.pdf",
    "data/report3.pdf"
]

documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# print("Total pages:", len(documents))

# for document in documents[:3]:
#     print(document.metadata)
#     print(document.page_content[:200])
#     print("----------------")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200           # help preserver context between chunks
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

average_chunk_size = sum(
    len(chunk.page_content) for chunk in chunks
) / len(chunks)

print("Average chunk size:", round(average_chunk_size, 2), "characters")

# print("Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
# print("Embedding deployment:", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
# print("API version:", os.getenv("AZURE_OPENAI_API_VERSION"))

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Create a FAISS vector store from the document chunks and embeddings


vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Vector store created!")


query = "What are the main findings about the impact of generative AI on software developer productivity?"

results = vectorstore.similarity_search(query, k=3)     # 3 for matching context

print("\nQuery:", query)
print("\nTop matching chunks:")

for i, result in enumerate(results):
    print(f"\n--- Chunk {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)


# GPT 5 mini generation

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# Combine the retrieved chunks into context
context = "\n\n".join(
    result.page_content for result in results
)

prompt = f"""

Answer the question using the provided context.
If the answer is not contained within the context, respond with:
"I do not have information about that."

Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print("\n--- GPT-5-mini Answer ---")
print(response.content)



