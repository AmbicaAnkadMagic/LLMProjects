from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from InstructorEmbedding import INSTRUCTOR

load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["GOOGLE_API_KEY"])

instructor_embeddings = HuggingFaceInstructEmbeddings ()

vecdb_file_path = "faiss_store"
def create_vector_db():
    loader = CSVLoader (file_path='codebasics_faqs.csv', source_column='prompt')
    docs = loader.load()
    vector_db=FAISS.from_documents(documents=docs, embedding=instructor_embeddings)
    vector_db.save_local(vecdb_file_path)

def get_qa_chain():
    #load local vector db
    vector_store = FAISS.load_local("faiss_store", HuggingFaceInstructEmbeddings(), allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever()
    rdocs = retriever.get_relevant_documents("for how long this course is valid")



    prompt_template = """ Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context. 
    If the anser is not found in the context, kindly state "I don't know." Don't try to make up answer.
    CONTEXT: {context}

    QUESTION: {question}
    """

    PROMPT = PromptTemplate (template= prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff", retriever=retriever, 
                                        input_key="query", return_source_documents=True,
                                       chain_type_kwargs={"prompt": PROMPT})
    return chain 

#response = chain.invoke("Do you know Dhaval's age?")

#if __name__ == "__main__":
 #   chain = get_qa_chain()
  #  print (chain ("Do you provide internship? Do you have EMI option?"))