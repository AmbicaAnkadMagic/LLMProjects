import json
import os
import sys
import boto3
import streamlit as st


from langchain_community.embeddings import BedrockEmbeddings

from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

## Vector Embeddings and Vector Stote 
from langchain.vectorstores import FAISS

##LLM Models 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


## Bedrock Clients 
bedrock=boto3.client(service_name="bedrock-runtime")

bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
## Data ingestion 
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents = loader.load()

    # character splitting works better with this pdf data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000,
                                                   chunk_overlap = 1000)
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embeddings and Vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm=Bedrock(model_id = "anthropic.claude-v2:1", client=bedrock,
                model_kwargs={'max_tokens_to_sample':512})
    return llm

def get_llama3_llm():
    llm=Bedrock(model_id = "meta.llama3-8b-instruct-v1:0", client=bedrock,
                model_kwargs={'max_gen_len':512})
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
    )
    answer=qa({"query":query})
    if 'result' in answer:
        return answer['result']
    else:
        return "No result found."


def main():
    st.set_page_config("Chat PDF")
    st.header('Chat with PDF using AWS Bedrock')
    user_question = st.text_input("Ask a Question from PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store: ")

        if st.button("Vactors Update"):
            with st.spinner("Processing"):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("LLama3 Output"):
        with st.spinner("Processing"):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()