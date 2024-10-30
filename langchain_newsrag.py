import os
import streamlit as st
from datetime import datetime
from typing import List, Dict
import requests
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import re

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables

dotenv.load_dotenv()

# Set environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["NEWSAPI_KEY"] = os.getenv("NEWS_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

class NewsCollector:
    def __init__(self):
        self.base_url = "https://newsapi.org/v2/top-headlines"
        self.api_key = os.getenv("NEWS_API_KEY")
        
    def get_news(self, country: str, category: str) -> List[Dict]:
        params = {
            "country": country,
            "category": category,
            "apiKey": self.api_key,
            "pageSize": 5
        }
        response = requests.get(self.base_url, params=params)
        return response.json()["articles"]
    
    def preprocess_news(self, articles: List[Dict], region: str, category: str) -> List[Document]:
        documents = []
        for article in articles:
            if article['title'] != "[Removed]" and article['description'] != "[Removed]" and article['content'] != "[Removed]":
                content = f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}"
                summary_content = f"Title: {article['title']}\nDescription: {article['description']}"
                metadata = {
                    "published_date": article["publishedAt"],
                    "category": category,
                    "region": region,
                    "source": article["source"]["name"],
                    "summary_content": summary_content  # Store concise version in metadata

                }
                documents.append(Document(page_content=content, metadata=metadata))
        return documents


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
    
def main():
    # Initialize session state for articles if it doesn't exist
    if 'all_articles' not in st.session_state:
        st.session_state.all_articles = []

    # Initialize news collector, summarizer, vectorstore, and custom retriever
    news_collector = NewsCollector()
    vector_store = VectorStore()
    news_summarizer = NewsSummarizer()
    current_date = datetime.now().strftime("%Y-%m-%d")


    # Main interface
    st.title("NewsRag: Your AI News Assistant")

    # Dropdown menu for selecting categories
    selected_categories = st.multiselect(
        "Select News Categories", 
        options=["business", "entertainment", "general", "sports", "technology"], 
        default=["general"],
        key="category_selector"
    )

    if not selected_categories:
        st.warning("Please select at least one category")
        return

    # Compact Example Prompts in Sidebar
    with st.sidebar:
        st.header("Example Prompts")
        example_prompts = [
            "What's the latest business news?",
            "US election 2024 updates",
            "What is Taylor Swift up to?",
            "What's the latest news in the tech sector?",
            "Current sports events in India",
            "Entertainment industry updates",
            "Global market trends today",
            "Recent political developments"
        ]
        st.markdown("Copy a prompt to ask your AI News Assistant:")
        st.markdown("---")
        for prompt in example_prompts:
            st.markdown(f"• {prompt}")

    # Functionality 1: Category-based News Summary
    st.header("📝 AI news summary")
    
    generate_summary = st.button(
        "🔄 Generate AI news summary for Selected Categories",
        disabled=not selected_categories
    )
    
    if generate_summary:
        try:
            with st.expander("Expand to view the summary", expanded=True):
                with st.spinner("Fetching news articles..."):
                    # Reset articles list in session state
                    st.session_state.all_articles = []
                    
                    # Collect and preprocess articles for selected categories
                    for category in selected_categories:
                        articles = news_collector.get_news(country="us", category=category)
                        st.session_state.all_articles.extend(
                            news_collector.preprocess_news(articles, region="us", category=category)
                        )
                    
                    if not st.session_state.all_articles:
                        st.error("No articles found for the selected categories. Please try again or select different categories.")
                        return
                    
                    # Display meta information with two-column layout
                    total_articles = len(st.session_state.all_articles)
                    col1, col2 = st.columns(2)
                    col1.subheader("Total Articles Retrieved:")
                    col1.write(f"{total_articles} articles")

                    col2.subheader("Category-wise Article Count:")
                    for category in selected_categories:
                        cat_articles = [a for a in st.session_state.all_articles if a.metadata['category'] == category]
                        col2.write(f"{category.capitalize()}: {len(cat_articles)} articles")
                    
                    # Summarize and display news
                    summary = news_summarizer.summarize_news(st.session_state.all_articles, current_date)
                    st.subheader("News Summary")
                    st.write(summary)
        except Exception as e:
            st.error(f"An error occurred while fetching news: {str(e)}")
            return

    # Functionality 2: Custom Prompt Handling
    st.header("📝 Got specific questions? Ask away!")
    user_query = st.text_input("Enter your custom prompt:", key="user_query")
    
    if user_query:
        if not st.session_state.all_articles:
            st.warning("Please generate the AI news summary first to initialize the news database.")
        else:
            try:
                with st.spinner("Retrieving and processing relevant articles..."):
                    vector_store_instance = vector_store.create_vectorstore(st.session_state.all_articles)
                    custom_retriever = CustomNewsRetriever(vector_store_instance)
                    similar_prompts = custom_retriever.generate_questions(user_query)
                    relevant_docs = custom_retriever.get_relevant_docs(similar_prompts)
                    
                    if not relevant_docs:
                        st.warning("No relevant articles found for your query. Try rephrasing or asking a different question.")
                        return
                    
                    # Display similar prompts generated and article count
                    with st.expander("Expand to view similar prompts", expanded=False):
                        st.subheader("Similar Prompts Generated")
                        for i, prompt in enumerate(similar_prompts, 1):
                            st.markdown(f"{i}. {prompt}")
                    
                    # Display relevant articles retrieved
                    with st.expander("Expand to view relevant articles", expanded=False):
                        st.subheader("Relevant Articles Retrieved")
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"{i}. {doc.metadata['source']}: {doc.metadata['published_date']}")
                    
                    # Generate and display response
                    response = custom_retriever.answer_query(user_query, relevant_docs, current_date)
                    st.subheader("Generated Response")
                    st.write(response)
            except Exception as e:
                st.error(f"An error occurred while processing your query: {str(e)}")

    # Footer
    st.markdown(
        "<footer style='text-align: center; padding: 10px; font-size: small;'>"
        "❤️ it? Connect with me on LinkedIn: "
        "<a href='https://www.linkedin.com/in/naraparajusaisandeep/' target='_blank'>Sai Sandeep Naraparaju</a>"
        "</footer>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()