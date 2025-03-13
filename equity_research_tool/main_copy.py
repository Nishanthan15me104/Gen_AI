
import os
import streamlit as st
import pickle
import time

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai  import GoogleGenerativeAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


st.title(" News Research tool 📈")
st.sidebar.title("News Article URLs")
main_placeholder = st.empty()


urls =[]

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked =  st.sidebar.button(" process URLS")
file_path = "faiss_store_googlepalm.pkl"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(google_api_key = GOOGLE_API_KEY, temperature = 0.2, model="models/gemini-2.0-flash" )

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls =urls)
    main_placeholder.text("data loading ...started ...")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("text splitter...started...")
    docs = text_splitter.split_documents(data)

    # create embeddings and sacve to faiss index
    api_key = os.getenv("GOOGLE_API_KEY")

   
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    vectorstore_googlepalm = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("embedding vector started....")
    time.sleep(2)

    # save the faiss idex to pickle file
    with open(file_path,'wb') as f:
        pickle.dump(vectorstore_googlepalm, f)
 
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
            result = chain({"question": query},return_only_outputs=True)
            #Answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

