import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    You are a AI document verifier for AICTE.
    Your task is to check if the input document submitted by the user : {context} is proper or not.
    You will use out these rules to verify a given document : {rules}.
    At the end you will provide a score which is percentage from 0 to 100%, which concludes how proper is user's document is.
    If the provided percentage of check is over 80% then the document is accepted, otherwise its rejected and in such a case of rejection, provide proper reasons for rejection, in bullet points and ask user to try again or do a manual verification appeal.
    You must use your own knowledge and the rules, to properly verify the provided context of document and try to automate the process of approval for AICTE.
    \n\n
    doc_name:\n {doc_name}?\n
    Rules: \n{rules}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["doc_name", "rules"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(prompt,doc_name):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(prompt)
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "doc_name": doc_name , "rules" : prompt}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


prompt1 = '''Rule 1 : 
Rule2 : 
Rule3:
    '''
prompt2 = '''Rule 1 : 
Rule2 : 
Rule3:
    '''
prompt3 = '''Rule 1 : 
Rule2 : 
Rule3:
    '''
prompt4 = '''Rule 1 : 
Rule2 : 
Rule3:
    '''

def main():
    st.set_page_config("AICTE Approver")
    st.header("Online document verification")

    choice1 = st.button("CRS Form")
    choice2 = st.button("Form B")
    choice3 = st.button("Form C")
    choice4 = st.button("Mandate Form")

    if  choice1:
        user_input(prompt1,"CRS Form")
    elif choice2:
        user_input(prompt2,"Form B")
    elif choice3:
        user_input(prompt3,"Form C")
    elif choice4:
        user_input(prompt4,"Mandate Form")

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