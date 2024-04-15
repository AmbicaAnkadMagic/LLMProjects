import numpy as np
import streamlit as st
import langchain_helper
st.title ("Restaurant Name Generator")
cuisine = st.sidebar.selectbox("Pick a Cuisine : ", ("Indian", "Arabic", "American", "Mexican", "Italian"))



if cuisine:
    response = langchain_helper.generate_restaurant_name_and_menu_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menuItems = response['menu_items'].strip().split(",")
    st.write('** Menu Items **')
    for i in menuItems:
        st.write(i)
