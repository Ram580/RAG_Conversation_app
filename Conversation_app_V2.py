# -*- coding: utf-8 -*-
"""RAG Conversational Chat Application using lanchain, Mistral 7B , Pinecone vector DB

### Step-1: Upload Documents and Load with Langchain Document Loader
- Upload the documents to Google Colab.
- Use Langchain document loader to load the documents.

### Step-2: Perform Chunking
- Perform chunking on the loaded documents.

### Step-3: Initialize LLM and Use Huggingface Embedding Model
- Initialize a Large Language Model (LLM).
- Use the Huggingface Embedding Model to convert the chunks into embeddings.

### Step-4: Initialize Vector Database
- Initialize a Vector Database to store the resulting embeddings.

### Step-5: Upload Embeddings to Vector Database
- Upload the embeddings to the Vector Database.

### Step-6: Create Langchain Conversational Buffer Memory
- Create a Langchain conversational buffer memory.

### Step-7: Create Prompt Template
- Create a prompt template for generating responses.

### Step-8: Use Langchain RetreivalQA
- Use Langchain RetreivalQA for creating the conversational chat.

### Step-9: Create Front End with Streamlit
- Create a front end for the application using Gradio.

### Step-10: Upload Code to GitHub
- Upload the code to a GitHub repository.

### Step-11: Deploy App in Huggingface Spaces
- Deploy the application in Huggingface Spaces.

### Step-12: Create Documentation
- Create documentation for the entire process followed.
"""

# Installing the required libraries
# !pip install langchain
# !pip install pypdf
# !pip install sentence-transformers==2.2.2
# !pip install pinecone-client==2.2.4
# !pip install unstructured
# !pip install "unstructured[pdf]"

# initializing the Huggingface API to access Embeddig models
# from google.colab import userdata
# HUGGINGFACE_API_KEY = userdata.get('Hugging_Face_API_Key')
# HUGGINGFACE_API_KEY=HUGGINGFACE_API_KEY

# Creating a directory to store the data

# from langchain.document_loaders import PyPDFDirectoryLoader

# loader = PyPDFDirectoryLoader("data")

# importing all the required Libraries
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# from langchain.document_loaders import PyPDFDirectoryLoader
# loader = PyPDFDirectoryLoader("data")
# data = loader.load()

# len(data)

import os

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# creating chunking for the above data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# # creating chunking for the above data
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
# chunked_data=text_splitter.split_text(data)

# Create Embeddings using Huggingface Embeddings
import sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Initializing Pinecone
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY', 'f7384d73-ea97-45ca-abaa-9b14327fd50f')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'gcp-starter')

import pinecone
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "pinecone-demo" # put in the name of your pinecone index here

from langchain.vectorstores import Pinecone

# Load the data into pinecone database
def get_vector_store(text_chunks):
   #docsearch = Pinecone.from_texts(chunked_data, embeddings, index_name=index_name)
   docsearch = Pinecone.from_texts([t for t in text_chunks], embeddings, index_name=index_name)
   return docsearch


# query = "How many topics are covered?"
# docs = docsearch.similarity_search(query, k=1)
# docs

from langchain import HuggingFaceHub

llm=HuggingFaceHub(huggingfacehub_api_token="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",repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")

# from langchain.chains import RetrievalQA
# # retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
# retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                   chain_type="stuff",
#                                   retriever=retriever,
#                                   return_source_documents=True)

#question = "What are the Technical  Skills to learn for a Promising AI Career?"

#print(qa_chain(question))

## Adding Memory component
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True, max_history_length=5
)


import streamlit as st


# Chat History
#chat = llm.start_chat(history=[])
# intialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def user_input(user_question):
    # Load embeddings only once (assuming same model for both)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Pinecone search using the loaded embeddings
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    docs = docsearch.similarity_search(user_question)

    # Define prompt template
    template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details,
                if the answer is not available in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
                {context}
                Donot provide the Context , Provide the Answer only , to the question in the following format
                Question: {question}
                Helpful Answer:"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    #prompt =  PromptTemplate(input_variables=["question"], template=template)
    
    # Create retriever and chain using the loaded embeddings
    retriever = docsearch.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=retriever,
        memory=memory
    )

    # Extract context from retrieved documents (replace with your logic)
    # Consider filtering or summarizing retrieved documents
    #context = " ".join([doc.get("text", "") for doc in docs[:3]])
    context = docs

    # Inject prompt into query (alternative approach)
    query = f"{template.format(context=context, question=user_question)}\nQuestion: {user_question}"
    #query = f"{template.format( question=user_question)}\nQuestion: {user_question}"
    
    response = qa_chain(
        {"question": query},
        return_only_outputs=True
    )

    # Display response
    #st.write("Reply: ", response["answer"])
    Ans = extract_helpful_answer(response)
    st.write(Ans)
    
    # Feature to load Chat history
    if st.button("Load Chat History"):
        # add user query and response to session chat history
        st.session_state['chat_history'].append(("you",user_question))
        # for chunk in Ans:
        #     #st.write(chunk.text)
        #     st.session_state['chat_history'].append(("AI Assistant",chunk))
        st.session_state['chat_history'].append(("AI Assistant",Ans))
        st.subheader("The chat history is ")
        for role,text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")
            
    # Feature to load Related Context from the uploaded Documents
    if st.button("Load Related Context from Your Document"):
        related_context = docs
        st.subheader("Related Context from Your Document:")
        for doc in related_context:
            st.write(f"Document: {doc}")
            st.write("\n")
    else:
        st.warning("Please enter a question before loading related context.")

def extract_helpful_answer(response):
    # Split the response by the delimiter "Helpful Answer:"
    parts = response["answer"].split("Helpful Answer:")

    # If there are two parts (before and after "Helpful Answer:"), return the second part
    return parts[2].strip()



      
def main():
    #st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Mistral")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

