
#########################################################

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PIL import Image  # Import the Image module from PIL for image processing
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import cv2
import numpy as np
import PIL.Image


# Set environment variables for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'count' not in st.session_state:
    st.session_state.count = 0

@st.cache_resource
def load_documents(directory):
    loader = DirectoryLoader(directory, loader_cls=TextLoader)
    text_documents = loader.load()
    return text_documents

@st.cache_resource
def split_documents(_documents, chunk_size=800, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(_documents)

@st.cache_resource
def create_embeddings(model_name):
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def create_db(_documents, _embeddings):
    return Chroma.from_documents(_documents, _embeddings)

#handle image with ocr with gemini
def OCR_text(image):
     img = PIL.Image.open(image)
     model = genai.GenerativeModel(model_name="gemini-1.5-pro")
     prompt = '''Work as a kind of ocr model, You are given an image of an architectural plan or a floor map of an college , the map can be of entities like a college main campus area or canteen or ground area etc.
     The provided image will have dimensions as well which are generally there in a map or might not be, you have to carefully interpret the given image and based on your interpretation as a     professional Architect with 20 year experience, so carefully first retrieve out all the text info , ie dimensions from the image to return in a structured manner.
     This returned information may also include info like the overview of the provided map and what does it signify etc. but the text info shall be clear and accurate, since this info will then be sent to a AI model for AI verification of the map based on the guidelines.
     '''
     response = model.generate_content([prompt, img])  
     return response.text 



def gemini_llm(question, context, raw_retrieved, connverse_history,ocr_text):
    history = f"{connverse_history}"
    if ocr_text != None:
        st.session_state.conversation_history.append(f"Ocr_text for image {st.session_state.count} was :{ocr_text}")

    print(history)
    prompt_template = """
    
    You are a great Architecture floor map/Document verification AI tool for AICTE,that helps users with the verification and approval of their Architectural floormaps based upon AICTE guidelines.
    You may refer to the context provided to you , for more specific and accurate details, the context provided to you is the details of how to verify a given document and what to check in it, please perform a through check accordingly from context : {context}.
    Otherwise use your own learnings to act as a official document verification tool.
    You must always first check the conversation history : {history} before answering anything , and make sure the conversations are continuous and make sense.
    Try you best to understand the context of the ocr_text : {ocr_text} , since that is the exact detail provided in user document that he wish to approve. 
    Your main task is as follows, check for the document submitted by user as ocr_text, based on rules and dimensions provided in context, try to verify it, if you feel that the document is correct and can be approved, reply with an official message saying so, also try to provide a approval or matching score to each document out of 100% , very carefully and accurately , only the documents with over 70% approval score based on proper verification can be approved. otherwise return proper reasons for disapproval and how to improve, in your response itself, if you disapprove a document , ask the user to request for a manual check, if he feel that his document is 100% correct and verified.
    Always start your response with a verification score in a very large size like "verification score : " and then move on to your response.
    Please return the dimensions received in ocr in a proper stuructured format at the end of your response to confirm user that his document or map is carefully interpreted.
    The response you provide will be directly shown to end user, so make sure to not include any key info and be very official with the details in you response and dont exponse your functioning.
Example Response Template:

VERIFICATION SCORE: XX%

Verdict: [Approved/Disapproved]
Reasons:
[Provide clear points for approval or disapproval based on the context.]
Suggestions (if applicable):
[Actionable suggestions to improve the document.]
Next Steps:
[Encourage a manual review request if needed.]
Extracted Dimensions:

[List extracted dimensions in a structured table format.]
    Context:\n{context}?\n
    Question:\n{question}\n
    history:\n{history}\n
    ocr_text:\n{ocr_text}\n
    
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history","ocr_text"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({
        "input_documents": raw_retrieved,
        "question": question,
        "history": history,
        "ocr_text": ocr_text    
    }, return_only_outputs=True)
    ai_response = response["output_text"]
    st.session_state.conversation_history.append(f"AI Response {st.session_state.count} was :{ai_response}")
    return response

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def rag_chain(question,retriever,ocr_text):
    imp_keywords = f'and the verification guidelines , proper set of rules to verify {question} , the given document , all the available guidelines to approve the provided document'    
    retrieved_docs = retriever.invoke(question + imp_keywords)
    raw_retrieved = retrieved_docs
    formatted_context = combine_docs(retrieved_docs)
    #formatted_context += charities_info
    #global converse_history
    #global count
    st.session_state.count += 1
    st.session_state.conversation_history.append(f"question {st.session_state.count} was :{question}")
    ans = gemini_llm(question, formatted_context, raw_retrieved, st.session_state.conversation_history,ocr_text)
    
    print(st.session_state.conversation_history[0:])
    return ans

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI approval", 
    page_icon="📃🤖", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
        /* Main background */
        .main {
            background: linear-gradient(to bottom right, #ffffff,#ffffff);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Button styles (smaller size) */
        .stButton button {
            background-color:#ccf7ff;
            color: #f55c20;
            font-size: 14px;  /* Reduced font size */
            padding: 8px 16px; /* Reduced padding */
            border: none;
            transition: background-color 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #9dff9c;
            color:white;
        }

        /* Text input styles with increased height */
        .stTextInput input {
            border-radius: 12px;
            padding: 12px;
            border: 2px solid #5c6bc0;
            background-color: #f0f4ff;
            color: #f55c20;
            height: 60px;  /* Increased height */
            width: 100%;
        }
        
        /* Title styling */
        .title h1 {
            color: #f55c20;
            font-size: 3.5rem;
            text-shadow: 1px 1px 2px #333;
            margin-bottom: 10px;
            font-family: oswald,sans-serif;

        }
        .title h4 {
            color: #cc5ac7;
            
        }


        /* Header styling */
        .header {
            background-color: #ffffff;
            color: #d3d3d3;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Prompt styles */
        .prompt {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
        }
        
        .prompt:hover {
            background-color: #ffffff;
        }

        /* Footer styling */
        .footer {
            font-size: 0.9rem;
            color: #777;
            text-align: center;
            margin-top: 20px;
        }

        /* Image styling */
        .stImage img {
            border: 2px solid #ff8a65;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown("<div class='title'><h1>College Architecture/Map verification</h1></div>", unsafe_allow_html=True)
st.markdown("""
    <div class="header">
        <h2>Hello there, This is a AI powered Architecture and floor map verification portal for AICTE.</h2>
        <h4>Please make sure you provide a clear floor map of the proposed feild with clear details on dimensions and proper division of space, which otherwise can lead to cancellation of the approval.</h4>
    </div>
    """, unsafe_allow_html=True)

# Prompt suggestions

prompts = {
    "Campus map/plan": "guidelines to verify the Campus map/plan of the college",
    "Parking Area map/plan": "guidelines to verify the Parking Area of the college",
    "Ground and sports area/plan": "guidelines to verify the Ground and sports area of the college",
    "stage dimensions/space": "guidelines to verify the stage dimensions of the college",
    "Canteen area/plan": "guidelines to verify the Canteen area of the college "
}


documents = load_documents("RagDB")
split_docs = split_documents(documents)
if split_docs:
    embeddings = create_embeddings("gemini-1.5-flash")
    if embeddings:
        db = create_db(split_docs, embeddings)
        retriever = db.as_retriever() if db else None
    else:
        st.error("Embeddings were not created successfully. Please check the embedding model configuration.")
else:
    st.error("No documents to split. Please check the document loading process.")


retriever = db.as_retriever()

# Session state for image and OCR text
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None


# Image uploader with session handling
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_image is not None:
    if uploaded_image != st.session_state.uploaded_image:
        st.session_state.uploaded_image = uploaded_image
        st.session_state.ocr_text = OCR_text(uploaded_image)  # Perform OCR only once and store result

# Display the uploaded image
if st.session_state.uploaded_image is not None:
    st.image(st.session_state.uploaded_image, caption="Uploaded Image", width=150)
    if st.button("Remove Image"):
        st.session_state.uploaded_image = None
        st.session_state.ocr_text = None
        st.experimental_rerun()  # Reload the page to clear the image display
if  st.session_state.uploaded_image is not None:
 st.markdown("<div class='title'><h3>Please select the type of document to be approved by AICTE:</h3></div>", unsafe_allow_html=True)
 for prompt in prompts:
    if st.button(prompt, key=prompt) :
        try:
            with st.spinner("Verification in Progress, Please wait..."):
                 response = rag_chain(prompts[prompt], retriever , st.session_state.ocr_text)
                 st.markdown(f"<p style='font-size: 2em;'>{response['output_text']}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"There is some issue at the moment: {e}. Please try again or reload!")
    elif(st.session_state.uploaded_image is None) :
        st.write(st.error("please upload a document first"))
else:
    st.write("Please upload a document that you wish to verify")
     


# Footer
st.markdown("<div class='footer'>Please be kind and avoid vulgar language. I'm here to help! 🤖</div>", unsafe_allow_html=True)
