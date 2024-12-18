# NewsRag: Your AI News Assistant 🤖📰

NewsRag is an intelligent news aggregation and analysis application that leverages the power of AI to provide personalized news summaries and answer specific queries about current events. Built with Streamlit and powered by LangChain, this application offers an interactive way to stay updated with the latest news across multiple categories.

## 🤖 Live Demo

[Click here](https://newsaisummarizerbot.streamlit.app/) to try it out!

## 🌟 Features

- **AI-Generated News Summaries**: Get concise summaries of news articles across different categories
- **Custom News Queries**: Ask specific questions about current events and receive targeted responses
- **Multi-Category Support**: Access news from various categories including:
  - Business
  - Entertainment
  - General
  - Sports
  - Technology
- **Smart Document Retrieval**: Utilizes vector similarity search to find relevant news articles
- **Dynamic Question Generation**: Automatically generates sub-questions to improve search relevance

## 🛠️ Technologies Used

- **Frontend**: 
  - Streamlit - For the interactive web interface
  
- **AI/ML Components**:
  - LangChain - For orchestrating AI workflows
  - LangSmith - For tracking LLM I/O
  - Groq - LLM provider for text generation
  - Google Generative AI - For document embeddings
  - Chroma DB - Vector store for similarity search
  
- **News Data**:
  - NewsAPI - For fetching real-time news data
  
- **Additional Libraries**:
  - python-dotenv - For environment variable management
  - pysqlite3 - For database operations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Groq
  - NewsAPI
  - Google AI
  - LangChain

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/newsrag.git
cd newsrag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```env
GROQ_API_KEY=your_groq_api_key
NEWS_API_KEY=your_newsapi_key
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run langchain_newsrag.py
```

2. Open your browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)

## 📱 Usage

1. **Generate News Summary**:
   - Select one or more news categories from the dropdown menu
   - Click "Generate AI news summary" to get a comprehensive summary of latest news (last one day)

2. **Ask Custom Questions**:
   - Enter your specific query in the text input field
   - View similar questions generated by the AI
   - Get relevant articles and a targeted response to your query

## 🏗️ Project Structure

```
newsrag/
├── langchain_newsrag.py    # Main Streamlit application
├── collect_news.py         # News collection functionality
├── custom_news.py          # Custom query handling
├── summarize_news.py       # News summarization logic
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables (create this)
```


## 👨‍💻 Author

Sai Sandeep Naraparaju - [LinkedIn](https://www.linkedin.com/in/naraparajusaisandeep/)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the AI orchestration framework
- [Streamlit](https://streamlit.io/) for the wonderful web framework
- [NewsAPI](https://newsapi.org/) for providing news data
- [Groq](https://groq.com/) for free access to LLM
- [Google](https://ai.google.dev/) for free access to embedding models

---
⭐ If you find this project helpful, please consider giving it a star!