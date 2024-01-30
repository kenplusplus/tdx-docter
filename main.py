import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import shutil
import argparse

PDF_PATH=os.path.join(os.path.dirname(__file__), "docs")

def load_pdfs():
    faiss_index_path = os.path.join(os.path.dirname(__file__), "faiss_index")

    if os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)

    pdfs = [f for f in os.listdir(PDF_PATH) if os.path.isfile(os.path.join(PDF_PATH, f))]

    text=""
    for pdf in pdfs:
        print("process PDF: %s..." % pdf)
        pdf_reader= PdfReader(os.path.join(PDF_PATH, pdf))
        for page in pdf_reader.pages:
            text+= page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    return text

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-b', '--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        load_pdfs()

    st.set_page_config("TDX Doctor")
    st.header("Please ask questions related to TDX or UEFI")

    user_question = st.text_input("Ask a Question like 'please describe xxx in 400 words'")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()