
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
#def OCR_text(image):
#     img = PIL.Image.open(image)
#     model = genai.GenerativeModel(model_name="gemini-1.5-pro")
 #    prompt = '''detect the text in this photo just like a ocr model? 
#     The text you are reading out is a document , a real document submitted by the user that he wish to verify
 #    it can be both digital as well as written , so make sure to take off every small detail off the document and in a structured manner if required , so that it can be verified by 
 #    our model in next step.
 #    Since you are approving documents , also make sure to specify in your ocr response if the given document has any image , QR or other hologram type identification in it, if so ,also try to tell what do you get from the image.
 #    '''
 #    response = model.generate_content([prompt, img])  
  #   return response.text 
def OCR_text(image):
      langs = ["en","hi"] # Replace with your languages - optional but recommended
      det_processor, det_model = load_det_processor(), load_det_model()
      rec_model, rec_processor = load_rec_model(), load_rec_processor()
      predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
      output = ""
      for ocr_result in predictions:
         for text_line in ocr_result.text_lines:
          output +=  text_line.text + "\n"  
          
           # Add a newline after each text segment
      return output

def detect_qr_and_images(pil_image):
    # Convert the PIL image to an OpenCV-compatible format (NumPy array)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Initialize the QRCode detector
    qr_detector = cv2.QRCodeDetector()
    qr_data, _, _ = qr_detector.detectAndDecode(image)

    # Check for QR code presence
    has_qr_code = bool(qr_data)  # True if QR data is detected

    # Detect potential image regions (non-text) based on contours
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Flag if large non-text regions (likely images or graphics) are detected
    has_image_content = any(cv2.contourArea(c) > 1000 for c in contours)  # Threshold area size for image content

    return has_qr_code, has_image_content


def gemini_llm(question, context, raw_retrieved, connverse_history,ocr_text,has_qr,has_image):
    history = f"{connverse_history}"
    if ocr_text != None:
        st.session_state.conversation_history.append(f"Ocr_text for image {st.session_state.count} was :{ocr_text}")

    print(history)
    prompt_template = """
    
    You are a great Document verification AI tool for AICTE,that helps users with the verification and approval of their official documents.
    You may refer to the context provided to you , for more specific and accurate details, the context provided to you is the details of how to verify a given document and what to check in it, please perform a through check accordingly from context : {context}.
    Otherwise use your own learnings to act as a official document verification tool.
    You must always first check the conversation history : {history} before answering anything , and make sure the conversations are continuous and make sense.
    Try you best to understand the context of the ocr_text : {ocr_text} , since that is the exact detail provided in user document that he wish to approve.
    make sure to consider the two value of has_qr:{has_qr} and has_image:{has_image} from the input that tells you if the given document has any valid image and qr or not that may help you to determine a better verification score of the document. 
    Your main task is as follows, check for the document submitted by user as ocr_text, based on rules provided in context, try to verify it, if you feel that the document is correct and can be approved, reply with an official message saying so, also try to provide a approval score to each document out of 100% , very carefully and accurately , only the documents with over 70% approval score based on proper verification can be approved. otherwise return proper reasons for disapproval and how to improve, in your response itself, if you disapprove a document , ask the user to request for a manual check, if he feel that his document is 100% correct and verified.
    Always start your response with a verification score in a very large size like "verification score : " and then move on to your response.
    Be very carefull in counting the number of digits , for documents like adhar card , do not count spaces as digits, the spaces are meant to separate 3 sections of adhar card of 4 digits each to a total of 12, so this means a valid adhar number will look something like this example: 2345 7819 9212 , with total 12 numbers separated with space.
    If the given ocr_text is off a official document , make sure to structure is down as name, unique id and all nessasary details of that document and return it at the end of your response with a proper heading and structure. 
    The response you provide will be directly shown to end user, so make sure to not include any key info and be very official with the details in you response and dont exponse your functioning.
    Context:\n{context}?\n
    Question:\n{question}\n
    history:\n{history}\n
    ocr_text:\n{ocr_text}\n
    has_qr: \n{has_qr}\n
    has_image: \n{has_image}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history","ocr_text","has_qr","has_image"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({
        "input_documents": raw_retrieved,
        "question": question,
        "history": history,
        "ocr_text": ocr_text,
        "has_qr": has_qr,
        "has_image": has_image
    }, return_only_outputs=True)
    ai_response = response["output_text"]
    st.session_state.conversation_history.append(f"AI Response {st.session_state.count} was :{ai_response}")
    return response

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def rag_chain(question,retriever,ocr_text,has_qr, has_image):
    imp_keywords = f'and the verification guidelines , proper set of rules to verify {question} , the given document , all the available guidelines to approve the provided document'    
    retrieved_docs = retriever.invoke(question + imp_keywords)
    raw_retrieved = retrieved_docs
    formatted_context = combine_docs(retrieved_docs)
    #formatted_context += charities_info
    #global converse_history
    #global count
    st.session_state.count += 1
    st.session_state.conversation_history.append(f"question {st.session_state.count} was :{question}")
    ans = gemini_llm(question, formatted_context, raw_retrieved, st.session_state.conversation_history,ocr_text,has_qr,has_image)
    
    print(st.session_state.conversation_history[0:])
    return ans

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Verification", 
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
st.markdown("<div class='title'><h1>Document Verification for AICTE</h1></div>", unsafe_allow_html=True)
st.markdown("""
    <div class="header">
        <h2>Hello there, This is a AI powered document verification portal for AICTE.</h2>
        <h4>Please submit the documents you wish to verify in clear quality and computerized format for smooth verification.</h4>
    </div>
    """, unsafe_allow_html=True)

# Prompt suggestions

prompts = {
    "Adhar_Card": "guidelines to verify the Adhar card",
    "College_id": "guidelines to verify the college card",
    "Pan_card": "guidelines to verify the Pan card",
    "Voter_id": "guidelines to verify the voter ID",
    "College_Guidelines": "guidelines to verify the college guidelines"
}


documents = load_documents("rag_feed")
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
  image = Image.open(uploaded_image)
  st.session_state.has_qr_code, st.session_state.has_image_content = detect_qr_and_images(image)

if uploaded_image is not None:
    if uploaded_image != st.session_state.uploaded_image:
        st.session_state.uploaded_image = uploaded_image
        st.session_state.ocr_text = OCR_text(image)  # Perform OCR only once and store result

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
                 response = rag_chain(prompts[prompt], retriever, st.session_state.ocr_text,st.session_state.has_qr_code,st.session_state.has_image_content)
                 st.markdown(f"<p style='font-size: 2em;'>{response['output_text']}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"There is some issue at the moment: {e}. Please try again or reload!")
    elif(st.session_state.uploaded_image is None) :
        st.write(st.error("please upload a document first"))
else:
    st.write("Please upload a document that you wish to verify")
     


# Footer
st.markdown("<div class='footer'>Please be kind and avoid vulgar language. I'm here to help! 🤖</div>", unsafe_allow_html=True)
