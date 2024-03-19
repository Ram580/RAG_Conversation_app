# RAG-powered Conversational Document Explorer
### Web APP link : https://ragconversationapp-benfydab7yhjxcz2jbhmhb.streamlit.app/

## Introduction:

This project showcases a cutting-edge Retrieval-Augmented Generation (RAG) application designed to empower users with intuitive, context-driven information retrieval from uploaded documents. By leveraging the power of Generative AI, this application seamlessly blends document understanding with natural language interaction.

## Key Features:

### Document Ingestion and Preprocessing:
- Effortlessly upload PDF documents. The application parses them using the PyPDF2 library, extracting the text content.
  
### Text Chunking:
- To optimize processing and storage efficiency, the extracted text is chunked into smaller segments using RecursiveCharacterTextSplitter from Langchain.
  
### Text Embedding with Sentence Transformers:
- Each text chunk is converted into a high-dimensional vector representation using Sentence Transformers' pre-trained all-MiniLM-L6-v2 model. This allows for efficient document similarity search.
  
### Pinecone Vector Database Integration:
- Pinecone, a high-performance vector database, is used to store the generated document embeddings. This enables fast retrieval of relevant documents based on user queries.
  
### Conversational Search with Memory:
- Users interact with the application through a Streamlit-built interface, asking questions in natural language.
- The application leverages a Conversational Retrieval Chain (CRC) to process user queries.
- The CRC utilizes a retriever component that searches Pinecone for documents similar to the user's question based on the text embeddings.
- A Langchain ConversationBufferMemory component maintains conversation history, allowing the application to consider previous interactions when responding to follow-up questions.
  
### Large Language Model Integration:
- The retrieved documents and conversation history are used to contextually tailor the response. The application employs the mistralai/Mixtral-8x7B-Instruct-v0.1 LLM from Hugging Face Open Source LLMs to generate informative answers to user questions.
  
### User-Friendly Interface:
- Streamlit provides a clean and intuitive interface for document upload, question input, and response display.

## Advanced Features:

### Load Chat History:
- Users can review the conversation history to stay on track and avoid repeating questions.
  
### Load Related Context from Your Document:
- Users can revisit excerpts from uploaded documents that directly relate to their current question, providing deeper context.

## Technical Deep Dive:

This project demonstrates proficiency in various technical aspects:

- **Natural Language Processing (NLP):**
  - Document parsing and text extraction from PDFs using PyPDF2
  - Text Chunking with RecursiveCharacterTextSplitter
  - Text Embedding with Sentence Transformers
  
- **Vector Database Integration:**
  - Utilizing Pinecone for efficient document similarity search.
  
- **Large Language Models (LLMs):**
  - Integrating and fine-tuning a pre-trained LLM (mistralai/Mixtral-8x7B-Instruct-v0.1) for context-aware response generation.
  
- **Conversational AI:**
  - Building a Conversational Retrieval Chain with memory capabilities using Langchain libraries.
  
- **Web Development Framework:**
  - Utilizing Streamlit for rapid development of a user-friendly web application interface.

## Benefits:

- **Efficient Document Search:**
  - Quickly access information from uploaded documents through natural language queries.
  
- **Conversational Exploration:**
  - Refine your understanding by engaging in a back-and-forth dialogue with the application.
  
- **Contextual Awareness:**
  - Uncover insights from documents tailored to your specific questions.
  
- **Improved Decision-Making:**
  - Gain the knowledge you need to make informed choices based on in-depth document analysis.
