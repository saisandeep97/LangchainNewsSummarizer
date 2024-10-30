import os
import streamlit as st
from datetime import datetime
import dotenv
from collect_news import NewsCollector
from summarize_news import NewsSummarizer, VectorStore
from custom_news import CustomNewsRetriever

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

# Load environment variables

dotenv.load_dotenv()

# Set environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["NEWSAPI_KEY"] = os.getenv("NEWS_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")


    
def main():
    # Initialize session state for articles if it doesn't exist
    if 'all_articles' not in st.session_state:
        st.session_state.all_articles = []

    # Initialize news collector, summarizer, vectorstore, and custom retriever
    news_collector = NewsCollector()
    vector_store = VectorStore()
    news_summarizer = NewsSummarizer()
    current_date = datetime.now().strftime("%Y-%m-%d")

    def format_datetime(date_str):
        """Format datetime string to a more readable format"""
        try:
            # Parse the ISO format datetime string
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
            # Format it to a more readable string
            # Example output: "Oct 29, 2024 - 3:53 AM"
            return dt.strftime("%b %d, %Y - %I:%M %p")
        except Exception:
            return date_str  # Return original string if parsing fails


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
            "Current sports events in US",
            "Entertainment industry updates",
            "Global market trends today",
            "Recent political developments"
        ]
        st.markdown("Copy a prompt to ask your AI News Assistant:")
        st.markdown("---")
        for prompt in example_prompts:
            st.markdown(f"{prompt}")

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
                            formatted_date = format_datetime(doc.metadata['published_date'])
                            st.markdown(f"{i}. {doc.metadata['source']}: {formatted_date}")
                    
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