# Krishi API

Krishi API is a FastAPI-based application designed to assist farmers and agricultural professionals by providing intelligent responses to queries and facilitating product searches. The API integrates advanced AI technologies to ensure accurate and efficient information retrieval.

## Features

- **Agricultural Query Resolution**: Uses a Retrieval-Augmented Generation (RAG) system to provide precise answers to farming-related questions.
- **Product Information Search**: Integrates Tavily Search for retrieving comprehensive details about agricultural products.
- **Dynamic Query Routing**: Implements a graph-based workflow to dynamically route queries for better efficiency.

## Technologies Used

- **FastAPI**: A high-performance web framework for building APIs.
- **Groq AI**: AI-driven responses for enhanced query resolution.
- **Hugging Face Embeddings**: Enables improved understanding of agricultural queries.
- **Tavily Search**: Provides robust product information retrieval capabilities.

## Installation

Follow these steps to set up and run Krishi API on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Prathamesh9284/krishi-api.git
cd krishi-api
```

### 2. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The API will be accessible at `http://127.0.0.1:8000`.

## Usage

Once the application is running, you can interact with the API using cURL, Postman, or directly through a web browser. FastAPI provides interactive documentation at:

- `http://127.0.0.1:8000/docs` (Swagger UI)
- `http://127.0.0.1:8000/redoc` (ReDoc UI)

## Configuration

Set up environment variables or a configuration file for key parameters such as:

- **API Keys**: Required for Groq AI, Hugging Face, and Tavily Search integrations.
- **Database Settings**: If applicable, configure database connection settings.
