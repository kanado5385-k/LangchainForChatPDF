# -*- coding: utf-8 -*-
#from langchain.llms import OpenAI
#llm = OpenAI(openai_api_key="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")

#result = llm.predict("hi!")
#print(result)


#chat_model = ChatOpenAI(openai_api_key="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")

#result = chat_model.predict("안녕")
#print(result)
from langchain.chat_models import ChatOpenAI
import os
import tempfile
from loguru import logger
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

st.title("🔎 ChatNEWS 당신에게 뉴스를 알려줍니다.")
    
def pdf_to_document(pdf_file):   
    loader = PyPDFLoader(pdf_file)
    documents = loader.load_and_split()
    return documents
def main():
    pdf_folder_path = "C:\\Users\\정보통신공학과\\Desktop\\빅카인즈공모전\\정통\\코드\\forPDF"

    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    if not pdf_files:
        st.warning("PDF 파일이 폴더에 없습니다.")
    else:
        for pdf_file in pdf_files:
            pages = pdf_to_document(pdf_file)

            if pages:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators="#",
                    chunk_size = 300,
                    chunk_overlap  = 0,
                    length_function = len,
                    is_separator_regex = False,
    )

                chunks = text_splitter.split_documents(pages)  
                #st.write(chunks)  #청크 확인위한 코드 streamlit run app.py

                # 여기서부터는 추출된 텍스트를 사용하여 다음 단계를 수행할 수 있습니다.
                embeddings_model = OpenAIEmbeddings(openai_api_key="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")
                db = FAISS.from_documents(chunks, embeddings_model)

                st.header("PDF에게 질문해보세요!!")
                question = st.text_input('질문을 입력하세요')

                if st.button('질문하기'):
                    with st.spinner('Wait for it...'):
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
                        result = qa_chain({"query": question})
                        st.write(result["result"])
                
            else:
                st.warning(f"{pdf_file}에서 텍스트를 추출할 수 없습니다.")

if __name__ == "__main__":
    main()

