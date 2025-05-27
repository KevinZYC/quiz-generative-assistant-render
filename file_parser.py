import os
import pdfplumber
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import nltk
nltk.download("punkt")

def extract_text_from_pdf(pdf_path, delete=False):
    if delete:
        try:
            os.remove(pdf_path)
        except Exception as e:
            print("error deleting file", e)
    
    pages_with_numbers = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages_with_numbers.append((i, text))
    return pages_with_numbers

def split_text(pages, chunk_size=1000, chunk_overlap=200):
    splitter = NLTKTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    documents = []

    for page_number, text in pages:
        raw_doc = Document(
            page_content=text,
            metadata={"page": page_number}
        )

        split_docs = splitter.split_documents([raw_doc])
        documents.extend(split_docs)
    
    for doc in documents[:10]:
        print(f"[Page {doc.metadata['page']}]")
        print(doc.page_content)
        print("-" * 30)
    return documents

def split_text_semantic(pages):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chunker = SemanticChunker(embeddings=embedding_model)

    documents = [Document(page_content=text, metadata={"page": page}) for page, text in pages]
    semantic_chunks = chunker.transform_documents(documents)

    for doc in semantic_chunks[:10]:
        print(f"[Page {doc.metadata['page']}]")
        print(doc.page_content)
        print("-" * 30)
    return semantic_chunks

def embed_text(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def query_knowledge_points(vectorstore, k):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.3
    )
    print("querying with k value of", k)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI tutor reading a technical textbook. Use LaTeX for any mathematic content
        The passages may come from multiple pages. When possible, include the page number of any extracted concept:
        ---------------------
        {context}
        ---------------------

        {question}
        """
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    question = "Extract and list all definitions, theorems, and knowledge points likely to be seen in a test," \
    "return them one per line, sorted by page number in increasing order" \
    "for each line return the following using latex when applicable: page number, the definition/theorem/knowledge point, " \
    "then a short definition/explanation in this exact format: page:definition/theorem/knowledge point:description" \
    "do not return any other text/phrases, do not use any other symbols."
    return chain.run(question)

def convert_to_json(rows, output_file="output.json"):
    """
    Takes a list of rows like ["5: Avogadro's number is...", ...]
    and saves them as a list of JSON objects in a file.
    """
    result = []

    for row in rows:
        # Try to extract the page number and content
        if ":" in row:
            try:
                page, content = row.split(":", 1)
                result.append({
                    "page": int(page.strip()),
                    "content": content.strip()
                })
            except ValueError:
                # Handle unexpected formats gracefully
                result.append({
                    "page": None,
                    "content": row.strip()
                })
        else:
            result.append({
                "page": None,
                "content": row.strip()
            })

    return json.dumps(result, indent=4)
