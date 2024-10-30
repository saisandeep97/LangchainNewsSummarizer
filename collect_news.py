import os
from typing import Dict, List
import requests
from langchain_core.documents import Document


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