from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma



class VectorStore:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model='models/text-embedding-004',
            model_type='retrieval_document'
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vectorstore

class NewsSummarizer:
    def __init__(self):
        self.llm = ChatGroq(model="llama3-8b-8192")
        
    def summarize_news(self, documents: List[Document], current_date: str) -> str:
        template = """Current date: {current_date}
        Based on the following news articles and their published dates, provide a comprehensive summary of the latest news:
        
        {context}
        
        Prioritize more recent news while maintaining coherence in the summary.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        context = "\n\n".join([
            f"Article ({doc.metadata['published_date']}):\n{doc.metadata['summary_content']}"
            for doc in documents
        ])
        
        return chain.invoke({
            "context": context,
            "current_date": current_date
        })