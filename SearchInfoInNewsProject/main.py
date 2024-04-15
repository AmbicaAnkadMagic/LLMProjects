import os 
import streamlit as st
import pickle
import time
import langchain
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv() # take environment variable from .env

st.title ("News Research Tool ")
st.sidebar.title ("News Articles URLs")

urls = []

for i in range (3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
#file_path = "faiss_store_openai.pkl"
folder = "faiss_store"
#create llm
llm= OpenAI(temperature=0.8, max_tokens=500)

main_placeholder = st.empty()

if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading started .....")
    data = loader.load()

    #split data 
    text_splitter = RecursiveCharacterTextSplitter (
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 1000
    )
    main_placeholder.text("Text Splitter started .....")
    docs = text_splitter.split_documents(data)

    #create embeddings 
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embedding Vector started building .....")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    #save the FAISS index to a pickle file
    #with open(file_path, "wb") as f:
       # pickle.dump(vectorstore_openai, f)

    #save the FAISS index to a file
    vectorstore_openai.save_local(folder)


query = main_placeholder.text_input("Question : ")
if query:
    if os.path.exists(folder):
        #load index file
        vector_store = FAISS.load_local("faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vector_store.as_retriever())

        response = chain.invoke({"question":query},return_only_outputs=True)
        #{"answer":" ", "sources": []}
        st.header("Answer")
        st.write(response["answer"])

        #Display the sources, if available
        sources = response.get ("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n") #split the sources by new line
            for source in sources_list:
                st.write(source)
