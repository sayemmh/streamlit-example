from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  

st.set_page_config(
    page_title="PolicyNavigator",
    page_icon="️⚕️",
    # layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


# vectors = getDocEmbeds("gpt4.pdf")
# qa = ChatVectorDBChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), vectors, return_source_documents=True)

async def main():

    async def storeDocEmbeds(file, filename):
    
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
        
        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)
        
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)
        
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)

        
    async def getDocEmbeds(file, filename):
        
        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)
        
        with open(filename + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
            
        return vectors
    

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]


    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []


    #Creating the chatbot interface
    st.title("Navigate My Insurance Policy")

    multi = ''' When choosing an insurance policy, insurers should give you access to a Summary of Benefits and Coverage and other documents that describe the nuances of a health insurance plan. Even after choosing a plan, unless you're an industry expert, it can be tough to understand what everything means.
    '''
    st.markdown(multi)


    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    uploaded_file = st.file_uploader("Upload benefits PDFs here", type="pdf")

    if uploaded_file is not None:

        with st.spinner("Processing..."):
        # Add your code here that needs to be executed
            uploaded_file.seek(0)
            file = uploaded_file.read()
            # pdf = PyPDF2.PdfFileReader()
            vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name)
            qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)

        st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding " + uploaded_file.name]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()





        with container:
            # with st.form(key='my_form', clear_on_submit=True):
                # user_input = st.text_input("Query:", placeholder="e.g: Summarize the paper in a few sentences", key='input')
                # submit_button = st.form_submit_button(label='Send')
        
            user_input = st.chat_input("Ask me something about your benefits")
            if user_input:
                    st.write(f"User has sent the following user_input: {user_input}")





            # if submit_button and user_input:
            if user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")


if __name__ == "__main__":
    asyncio.run(main())