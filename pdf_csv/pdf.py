import openai
import os
import sys
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
import tempfile
import streamlit as st

from llm.llm import get_llm, get_llm_type

def create_temp_file(uploaded_file):
    '''Function `create_temp_file(uploaded_file)` creates a temporary file and writes the uploaded content to it. Returns the name of the temporary file.'''
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # Write the uploaded content to the temporary file
    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name

def get_embedding_model():
    '''Function `get_embedding_model()` returns the embedding model.'''
    # embedding model
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}
    model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
    )  
    return model_norm

def pdf_logic(uploaded_files):
    '''Function `pdf_logic(uploaded_files)` processes the uploaded PDF files, splits the documents into chunks, creates an embedding model, saves it to a local file, and returns a Conversational Retrieval Chain.'''
    loaders = []
    file_name = []

    for file in uploaded_files:
        temp_file_path = create_temp_file(file)
        file_name.append(file.name)

        loaders.append(PyPDFLoader(temp_file_path))

    documents, metadatas = [], []

    for i, loader in enumerate(loaders):
        st.session_state.error_file = file.name
        documents.extend(loader.load())

        empty_files = False

        for value in documents:
            if len(value.page_content) < 1 and not empty_files:
                empty_files = True
            else:
                empty_files = False

            value.metadata["source"] = file_name[i]
        
        if empty_files:
            raise Exception('EmptyFileError')

    st.session_state.error_file = ""
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    docs = text_splitter.split_documents(documents)

    model_norm = get_embedding_model()
    
    db = FAISS.from_documents(docs, embedding=model_norm)
    db.save_local("./faiss")
    
    k = 4

    db = FAISS.load_local("./faiss", model_norm)

     # llm
    llm = get_llm()
    type = get_llm_type()
    # Prompt
    if type == 'TogetherLLM':
        
        prompt_template = """
        [INST] <<SYS>>
        Based on the provided context, your task is to answer the question at the end using the given pieces of context. 
        If you do not know the answer, please state that you do not know instead of making up an answer. For questions that 
        do not have relevant context available, please state that the context is not available. Whenever possible, 
        please provide a detailed answer that utilizes the available context to provide an accurate response.
        The context is as follows:
        {context}
        <</SYS>>
        {question}[/INST]
        """
    elif type == 'ChatOpenAI':
        prompt_template ="""You are an assistant for question-answering tasks. You will be given a question and context.
Use the following context to answer the question.
Follow the below guidelines:
1.Understand the question and its meaning.
2.Understand the semantic meaning of the context and what is it trying to convey.
3.Based on the meanings of question and the context, if both the intents match, generate an answer to the question which is completely based on the context.
If the answer is not present in the context please say don't know the answer, dont generate the answer on your own . Give the detailed answer including all the important points from the context provided.
  Context: {context}
    Question: {question}
    Answer :"""
        
    PROMPT = PromptTemplate(template = prompt_template, input_variables=['context','question'])
    
    # chain
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                return_generated_question=True,
            )
    return qa

def get_docs_with_scores(prompt):
    '''Function `get_docs_with_scores(prompt)` returns the documents and their scores based on a given prompt.'''

    model_norm = get_embedding_model()
    db = FAISS.load_local("./faiss", model_norm)
    docs_and_scores = db.similarity_search_with_score(prompt)
    return docs_and_scores

def text_logic(uploaded_files):
    '''Function `text_logic(uploaded_files)` processes the uploaded text files, splits the documents into chunks, creates an embedding model, saves it to a local file, and returns a Conversational Retrieval Chain.'''
    loaders = []
    for file in uploaded_files:
        temp_file_path = create_temp_file(file)
        loaders.append(TextLoader(temp_file_path))

        
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    model_norm = get_embedding_model()   

    
    db = FAISS.from_documents(docs, embedding=model_norm)
    db.save_local("./faiss")

    db = FAISS.load_local("./faiss", model_norm)

    # llm
    llm = get_llm()
    type = get_llm_type()
    # Prompt
    if type == 'TogetherLLM':      
        prompt_template = """
        [INST] <<SYS>>
        Based on the provided context, your task is to answer the question at the end using the given pieces of context. 
        If you do not know the answer, please state that you do not know instead of making up an answer. For questions that 
        do not have relevant context available, please state that the context is not available. Whenever possible, 
        please provide a detailed answer that utilizes the available context to provide an accurate response.
        The context is as follows:
        {context}
        <</SYS>>
        {question}[/INST]
        """
    elif type == 'ChatOpenAI':
        prompt_template ="""You are an assistant for question-answering tasks. You will be given a question and context.
Use the following context to answer the question.
Follow the below guidelines:
1.Understand the question and its meaning.
2.Understand the semantic meaning of the context and what is it trying to convey.
3.Based on the meanings of question and the context, if both the intents match, generate an answer to the question which is completely based on the context.
If the answer is not present in the context please say don't know the answer, dont generate the answer on your own . Give the detailed answer including all the important points from the context provided.
  Context: {context}
    Question: {question}
    Answer :"""
        
    PROMPT = PromptTemplate(template = prompt_template,input_variables=['context','question'])
    chain_type_kwargs = {'prompt': PROMPT}

    # chain
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                return_generated_question=True,
            )
    return qa
