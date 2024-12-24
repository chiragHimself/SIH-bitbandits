from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from google.generativeai import genai
import cv2
import numpy as np
import os

# FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

genai_api_key = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key"

genai.configure(api_key=genai_api_key)

# Load models and processors for OCR
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

@app.post("/verify-document/")
async def verify_document(
    question: str = Form(...),
    context_directory: str = Form(...),
    image: UploadFile = File(None)
):
    # Load context documents
    loader = DirectoryLoader(context_directory, loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings and Chroma DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(split_docs, embeddings)
    retriever = db.as_retriever()

    ocr_text = None
    has_qr = False
    has_image = False

    if image:
        pil_image = Image.open(image.file)
        langs = ["en", "hi"]
        predictions = run_ocr([pil_image], [langs], det_model, det_processor, rec_model, rec_processor)

        ocr_text = "\n".join(
            text_line.text for ocr_result in predictions for text_line in ocr_result.text_lines
        )

        image_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Check for QR code
        qr_detector = cv2.QRCodeDetector()
        qr_data, _, _ = qr_detector.detectAndDecode(image_np)
        has_qr = bool(qr_data)

        # Detect image regions (non-text)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_image = any(cv2.contourArea(c) > 1000 for c in contours)

    # Define the prompt and LLM chain
    def gemini_llm(question, context, ocr_text, has_qr, has_image):
        prompt_template = """
        You are a Document verification AI for AICTE. Based on the provided context, OCR text, QR code, and image presence,
        verify the document and provide an official approval or rejection message with reasons. Include a verification score out of 100%.
        Context: {context}
        OCR Text: {ocr_text}
        QR Code Present: {has_qr}
        Image Content Present: {has_image}
        Question: {question}
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "ocr_text", "has_qr", "has_image", "question"]
        )
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        response = chain(
            {
                "context": context,
                "ocr_text": ocr_text,
                "has_qr": has_qr,
                "has_image": has_image,
                "question": question,
            },
            return_only_outputs=True
        )
        return response["output_text"]

    formatted_context = "\n\n".join(doc.page_content for doc in split_docs)
    response = gemini_llm(question, formatted_context, ocr_text, has_qr, has_image)

    return {
        "verification_result": response,
        "ocr_text": ocr_text,
        "has_qr": has_qr,
        "has_image": has_image,
    }
