import os
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"]="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"

llm = OpenAI(temperature=0.7)

def generate_restaurant_name_and_menu_items (cuisine):
    prompt_tmplt = PromptTemplate(input_variables=['cuisine'],
                       template= "I want to open a restaurant for {cuisine} food. Suggest a fancy name")

    name_chain = LLMChain (llm = llm, prompt=prompt_tmplt, output_key="restaurant_name")


    prompt_tmplt1 = PromptTemplate(input_variables=['restaurant_name'],
                       template= "Please suggest some menu items for {restaurant_name}. Return it as comma-separated list.")

    menu_chain = LLMChain (llm = llm, prompt=prompt_tmplt1, output_key="menu_items")


    out_chains = SequentialChain (chains
    = [name_chain, menu_chain], input_variables=['cuisine'],
    output_variables = ['restaurant_name', 'menu_items']
    )
    response = out_chains.invoke({'cuisine' : cuisine})


    return response


#if __name__ == "__main__":
    #print (generate_restaurant_name_and_menu_items("Indian"))