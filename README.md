# PDF RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that loads multiple PDF documents, splits them into chunks, generates embeddings using Azure OpenAI, stores them in a FAISS vector database, retrieves relevant document chunks, and uses GPT-5-mini to generate an answer based on the retrieved context.

## Project Overview

This is a RAG pipeline that uses three PDF reports as the knowledge base.

The pipeline:

1. Loads PDF documents using LangChain's `PyPDFLoader`
2. Splits the documents into smaller chunks using `RecursiveCharacterTextSplitter`
3. Generates vector embeddings using Azure OpenAI
4. Stores the embeddings in a FAISS vector database
5. Retrieves the most relevant chunks using semantic similarity search
6. Passes the retrieved context to GPT-5-mini
7. Generates an answer based only on the retrieved document context

## Technologies

* Python
* LangChain
* Azure OpenAI
* `text-embedding-3-small`
* GPT-5-mini
* FAISS
* PyPDF

## Project Structure

```text
Proj_RAG/
├── data/
│   ├── report1.pdf
│   ├── report2.pdf
│   └── report3.pdf
├── rag_pipeline.py
├── .gitignore
└── README.md
```


## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -U langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv pypdf
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment
AZURE_OPENAI_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_API_VERSION=your_api_version
```

The `.env` file should **not** be committed to GitHub.

## Running the Pipeline

From the project directory:

```bash
python3 rag_pipeline.py
```

The program loads the PDFs, creates the chunks and FAISS vector store, performs a similarity search, and sends the retrieved context to GPT-5-mini.

## Chunking

The documents are split using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

The 200-character overlap helps preserve context between neighboring chunks.

## Retrieval

The pipeline uses FAISS similarity search to retrieve the three most relevant chunks for a given query:

```python
results = vectorstore.similarity_search(query, k=3)
```

The retrieved chunks are then combined and provided to GPT-5-mini as context.

## Dataset and Metrics

The current dataset contains three PDF reports.

The pipeline reports:

* Total pages: **81**
* Total chunks: **263**
* Average chunk size: **845.65 characters**

These metrics are printed when the pipeline is executed.

## Example Query

```text
What are the main findings about the impact of generative AI on software developer productivity?
```

The query is embedded and compared against the FAISS vector store. The three most relevant chunks are retrieved and passed to GPT-5-mini.

## RAG Prompt

GPT-5-mini is instructed to answer using only the retrieved context. If the answer cannot be found in the retrieved documents, the model is instructed to state that the information could not be found in the provided documents.

## Limitations

* Retrieval quality depends on the selected chunk size and overlap.
* Only the top three retrieved chunks are provided to the language model.
* The system does not currently use a reranker.
* Answers are limited to information contained in the retrieved context and may miss relevant information if the correct chunk is not retrieved.
* The current implementation performs retrieval and generation for a single query at a time.

## Future Improvements

Possible improvements include:

* Experimenting with different chunk sizes and overlap values
* Comparing different embedding models
* Adding metadata-based filtering
* Testing different values of `k`
* Adding a reranking step
* Evaluating retrieval accuracy using a set of predefined questions and expected source passages
* Building a simple user interface for interactive document queries

## Author

Prisha Bobde
