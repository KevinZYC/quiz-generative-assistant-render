from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
import json
from file_parser import *
from fpdf import fpdf
from pdf_quiz_generator import PDFQuizGenerator

class ChatMemoryHandler:
    def __init__(self):
        self.chat_history = []
        self.vectorstore = None
        self.summary_json = None
        self.chain = None

    def load_pdf(self, path, delete=True, k=None):
        self.vectorstore = None
        pages = extract_text_from_pdf(path, delete)
        chunks = split_text(pages)
        self.vectorstore = embed_text(chunks)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k or len(chunks)})

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

        # Step 1: create history aware retriever
        rewrite_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            prompt=rewrite_prompt
        )

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")  # a prebuilt QA prompt
        combine_docs_chain = create_stuff_documents_chain(
            llm, retrieval_qa_chat_prompt
        )

        self.chain = create_retrieval_chain(
            history_aware_retriever, combine_docs_chain
        )

        data = query_knowledge_points(self.vectorstore, k or len(chunks))
        rows = data.split("\n")
        self.summary_json = convert_to_json(rows)

    def ask(self, message):
        if self.chain is None:
            raise Exception("PDF not loaded yet!")

        inputs = {"input": message, "chat_history": self.chat_history}
        response = self.chain.invoke(inputs)
        response_str = response["answer"] if isinstance(response, dict) else response

        self.chat_history.append(("human", message))
        self.chat_history.append(("assistant", response_str))

        return response_str
    
    def generate_quiz(self, output_path, num_mc=5, num_fr=3):
        if not self.summary_json:
            raise ValueError("No summary data available. Load a PDF first.")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.5
        )

        quiz_prompt = f"""
        Based on the following textbook summary, generate a practice quiz with {num_mc} multiple-choice questions,
        then {num_fr} free-response questions.

        This following textbook summary is a list of topics covered by the textbook with a description. Please test these topics exactly
        {json.dumps(self.summary_json, indent=2)}

        Each multiple choice question should have 4 choices, each free response question should test the application of the concept
        
        Return the answer key in a separate section after all the questions.

        If there are any mathematical content, return using LaTeX
         
        Do not return anything other than the question and answers, not even section headers.
        """

        response = llm.invoke(quiz_prompt).content

        pdf = PDFQuizGenerator()
        pdf.add_quiz(response)

        output_path = "quiz_output.pdf"
        pdf.output(output_path)
        return output_path