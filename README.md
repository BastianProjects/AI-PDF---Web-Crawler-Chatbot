The AI PDF & Web Crawler Chatbot is an intelligent application designed to assist users in extracting, indexing, and querying information from both uploaded PDF documents and web pages. It leverages advanced natural language processing (NLP) and vector search technologies to provide concise and accurate answers to user queries based on the indexed content.
Key Features:
1.	PDF Upload and Processing:
•	Users can upload multiple PDF files.
•	Extracts text and images from PDFs using high-resolution partitioning.
•	Summarizes the content of uploaded PDFs for quick insights.
2.	Web Page Crawling:
•	Allows users to input URLs to crawl and index web page content.
•	Processes and splits the content into manageable chunks for efficient querying.
3.	Vector-Based Search:
•	Uses a vector store (Chroma) to index and retrieve documents based on semantic similarity.
•	Supports filtering results by specific sources (e.g., individual PDFs or URLs) or querying across all indexed sources.
4.	Interactive Chat Interface:
•	Users can ask questions in a chat-like interface.
•	Provides concise answers based on the retrieved context from indexed sources.
•	Maintains chat history for context-aware responses.
5.	Customizable Models:
•	Utilizes OllamaEmbeddings for embedding generation and OllamaLLM for language model-based reasoning.
•	Configurable models for embeddings and language generation.
6.	Streamlit-Powered UI:
•	User-friendly interface built with Streamlit.
•	Sidebar options for managing indexed sources, clearing the vector database, and resetting chat history.
Technical Highlights:
•	Text Splitting: Uses RecursiveCharacterTextSplitter to split large documents into smaller, overlapping chunks for better indexing and retrieval.
•	PDF Partitioning: Employs partition_pdf for extracting text and images from PDFs, with support for high-resolution strategies.
•	Chat Prompt Template: Custom prompt templates ensure concise and context-aware answers.
•	Session Management: Maintains chat history and indexed data across user interactions.
Use Cases:
•	Research Assistance: Quickly retrieve answers from academic papers or web articles.
•	Document Summarization: Summarize lengthy PDFs for easier understanding.
•	Knowledge Management: Index and query information from multiple sources in one place.
This project is ideal for users who need a powerful tool to process and query large amounts of textual data from diverse sources efficiently.
