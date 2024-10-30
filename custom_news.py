from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re


class CustomNewsRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatGroq(model="llama3-8b-8192")
    
    def generate_questions(self, user_query: str) -> List[str]:
        template = """You are an AI language model assistant. Your task is to generate 3 different sub questions OR alternate versions of the given user question to retrieve relevant documents from a vector database.

        By generating multiple versions of the user question,
        your goal is to help the user overcome some of the limitations
        of distance-based similarity search.

        By generating sub questions, you can break down questions that refer to multiple concepts into distinct questions. This will help you get the relevant documents for constructing a final answer

        If multiple concepts are present in the question, you should break into sub questions, with one question for each concept

        Provide these alternative questions separated by newlines between XML tags. For example:

        <questions>
        - Question 1
        - Question 2
        - Question 3
        </questions>

        Original question: {question}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": user_query})
        result = re.findall(r'- (.+\?)', result)
        return result
    
    def deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """
        Deduplicate documents based on their content and metadata
        """
        unique_docs = []
        seen_contents = set()
        
        for doc in docs:
            # Create a unique identifier using content and published date
            content_identifier = (
                doc.page_content,
                doc.metadata.get('published_date', '')
            )
            
            if content_identifier not in seen_contents:
                seen_contents.add(content_identifier)
                unique_docs.append(doc)
        
        return unique_docs
    
    def get_relevant_docs(self, questions: List[str]) -> List[Document]:
        all_docs = []
        for question in questions:
            docs = self.vectorstore.similarity_search(question)
            all_docs.extend(docs)
        
        # Deduplicate documents
        return self.deduplicate_docs(all_docs)
    
    def answer_query(self, user_query: str, docs: List[Document], current_date: str) -> str:
        template = """Current date: {current_date}
        Based on the following news articles and their published dates, answer this question: {question}
        
        Context:
        {context}
        
        Provide a clear and focused answer based on the most recent and relevant information.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        # Use full content for detailed answers to specific queries
        context = "\n\n".join([
            f"Article ({doc.metadata['published_date']}):\n{doc.page_content}"
            for doc in docs
        ])
        
        return chain.invoke({
            "context": context,
            "question": user_query,
            "current_date": current_date
        })
    
    def answer_query(self, user_query: str, docs: List[Document], current_date: str) -> str:
        template = """Current date: {current_date}
        Based on the following news articles and their published dates, answer this question: {question}
        
        Context:
        {context}
        
        Provide a clear and focused answer based on the most recent and relevant information.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        # Use full content for detailed answers to specific queries
        context = "\n\n".join([
            f"Article ({doc.metadata['published_date']}):\n{doc.page_content}"
            for doc in docs
        ])
        
        return chain.invoke({
            "context": context,
            "question": user_query,
            "current_date": current_date
        })