import streamlit as st
from  langchain_helper import get_few_shot_db_chain


st.title("Ambs T Shirts : Database Q & A")
query = st.text_input("Enter your Quetion")

if query:
    chain = get_few_shot_db_chain()
    response = chain.invoke(query)
    st.header("Answer : ")
    st.write(response)