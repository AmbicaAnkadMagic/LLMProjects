import streamlit as st
from langchain_helper import create_vector_db
from langchain_helper import get_qa_chain

st.title ("Question & Answer from CSV datafile")
btn = st.button ("Create KnowledgeBase(this is for DS when then want to re-create vector DB) ")
if (btn):
    create_vector_db()

question = st.text_input("Enter your Q :")

if (question ):
    chain = get_qa_chain()
    response = chain(question)
    st.header("Anser :")
    st.write(response["result"])