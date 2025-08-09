# 📄 General PDF RAG Chatbot

A **production-ready, domain-agnostic RAG (Retrieval-Augmented Generation) chatbot** built with **Streamlit**, **LangChain**, and **Google Gemini AI**. This application enables intelligent document interaction through semantic search and natural language querying.

## 🚀 Features

### 📚 Core Functionality
- **PDF Document Upload** - Support for multiple PDF files
- **Semantic Search** - Advanced content retrieval using AI embeddings
- **Natural Language Queries** - Ask questions in plain English
- **Context-Aware Answers** - Responses based on document content
- **Document Summarization** - Generate summaries with metadata filtering
- **Conversation Memory** - Maintains chat history throughout sessions

### 🏗️ Architecture Highlights

#### **Level 1: Basic RAG Implementation**
- PDF parsing powered by **PyMuPDF**
- Text chunking and embedding with **Google Generative AI**
- Vector similarity search using **FAISS**

#### **Level 2: Modular Pipeline Design**
- **LangChain-style components** for flexible configuration
- Scalable and maintainable architecture

#### **Level 3: Conversational Intelligence**
- Persistent conversation history
- **LangChain message objects** for structured dialogue management

#### **Level 4: Advanced Metadata Management**
- Automatic tracking of **document names** and **page numbers**
- Smart filtering capabilities by document source
- Enhanced search precision

#### **Level 5: Agent-like Orchestration**
- Intelligent rule-based routing system:
  - **Question Answering** - Document-specific responses
  - **Summarization** - Content synthesis and overview
  - **Fallback Mode** - General knowledge when documents don't contain answers

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Google Gemini API, LangChain
- **Vector Database**: FAISS
- **PDF Processing**: PyMuPDF
- **Language**: Python 3.10+

## ⚡ Quick Start

### Prerequisites

- **Python 3.10+** (recommended)
- **Google Gemini API Key** 

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbhiTyagi012/rag_chatbot.git
   cd rag_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```
    "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the app**
   - Open your browser and navigate to `http://localhost:8501`


## 🎯 Usage

1. **Upload PDFs**: Use the sidebar to upload one or more PDF documents
2. **Ask Questions**: Type natural language questions about your documents
3. **Get Answers**: Receive contextual responses with source citations
4. **Summarize**: Request document summaries with optional filtering

## 🏷️ Example Queries

- *"What are the main topics covered in this document?"*
- *"Summarize the conclusions from the research paper"*
- *"Find information about machine learning algorithms"*
- *"What does page 5 say about data preprocessing?"*

## 🔧 Configuration

The application supports various configuration options:

- **Chunk Size**: Adjustable text chunking parameters
- **Similarity Threshold**: Control retrieval sensitivity
- **Memory Length**: Configure conversation history retention
- **Model Parameters**: Temperature, max tokens, etc.
