from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from file_parser import *
import json

'''
todo:
load pdf works, make chat function work
figure out how to query for best matches
figure out how to get memory to work without wasting too many tokens
'''

class ChatMemoryHandler:
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.vectorstore = None
        self.summary_json = None
        prompt = PromptTemplate(
            input_variables=["history", "input", "context"],
            template=""" 
            You are a helpful tutor. Remember the conversation so far:
            {history}

            Context from textbook:
            {context}

            User: {input}
            Tutor:"""
        )
        self.chain = LLMChain(
            llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite"),
            prompt=prompt
        )

    def load_pdf(self, path, delete=True, k=None):
        self.vectorstore = None
        pages = extract_text_from_pdf(path, delete)
        chunks = split_text(pages)
        self.vectorstore = embed_text(chunks)
        if not k:
            k = len(chunks)
        data = query_knowledge_points(self.vectorstore, k)
        rows = data.split("\n")
        self.summary_json = convert_to_json(rows)

    def ask(self, message, k=5):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.3
        )
        print("querying with k value of", k)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

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
        return chain.run(message)