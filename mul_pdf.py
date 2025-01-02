from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st
import base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-color : #B0D8F3;
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./background.png')
st.title("Blue Bean - Scholarship Finder")
input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  // This changes the text color inside the input box
}
div[data-baseweb="base-input"] {
    background-color: transparent;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)
loader = PyMuPDFLoader("./PDF/3061_G.pdf")

docs = loader.load()
doc = docs[0]

import os

pdfs = []
for root, dirs, files in os.walk('PDF'):
    # print(root, dirs, files)
    for file in files:
        if file.endswith('.pdf'):

            pdfs.append(os.path.join(root, file))
#print(len(pdfs))
docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()

    docs.extend(pages)
#print(len(docs))
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = text_splitter.split_documents(docs)
#print(len(docs)) 
#print(len(chunks))
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
#print(len(encoding.encode(docs[0].page_content)))
#print(len(encoding.encode(chunks[0].page_content)))
from langchain_ollama import OllamaEmbeddings

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

single_vector = embeddings.embed_query("this is some text data")
#print(len(single_vector))
index = faiss.IndexFlatL2(len(single_vector))
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
#print(len(chunks))
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 3,  'fetch_k': 100, 'lambda_mult': 1})
#docs = retriever.invoke(question)

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
prompt = hub.pull("rlm/rag-prompt")
prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Answers must be only with respect to India. 
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only. 
    If someone asks who created you say 'Barkha, Jiya and Sindhu made me'
    Question: {question} 
    Context: {context} 
    Answer:
"""




prompt = ChatPromptTemplate.from_template(prompt)
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# print(format_docs(docs))

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []





def generate_response(input_text):
    model="llama3.2"
    rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
    )
    model = ChatOllama(model="llama3.2", base_url="http://localhost:11434/")

    response = model.invoke(input_text)

    return response.content

with st.form("llm-form"):
    text = st.text_area("Hi! I'm Blue Bean, your go-to buddy for a blue-free day! Ask me any of your scholarship doubts.")
    submit = st.form_submit_button("Submit")
 
with st.popover("Frequently Asked Questions"):
    st.markdown("1. Who is eligible for national scholarships?")
    st.markdown("2. Can I apply for multiple national scholarships?")
    st.markdown("3. Can international students apply for national scholarships?")
    st.markdown("4. what are the scholarships available for engineering students?")
    st.markdown("5. How will I be notified if I receive a national scholarship?")
    st.markdown("6. What are the benefits of receiving a national scholarship?")
    st.markdown("7. What is the selection process for a national scholarship?")
    st.markdown("8. More about PRAGATI SCHOLARSHIP SCHEME")
    st.markdown("9. More about SAKSHAM SCHOLARSHIP SCHEME")
    st.markdown("10. More about SWANATH SCHOLARSHIP SCHEME")
    st.markdown("11. More about Students of N.E.R for Higher Professional Courses")
    st.markdown("12. What are the benefits of receiving a national scholarship?")
    st.markdown("13. What documents are typically required for a national scholarship?")
    st.markdown("14. How do I apply for a national scholarship?")
    st.markdown("15. What are the scholarships available for diploma students ?")
    st.markdown("16. What are the scholarships available for 11 and 12 grade students ?")
    st.markdown("17. How can I know if I qualify for a national scholarship?")
    st.markdown("18. Is there an age limit for national scholarships?")
    st.markdown("19. Can I apply for a national scholarship if I am already receiving other financial aid?")
    st.markdown("20. Can I apply for a national scholarship as a part-time student?")
    st.markdown("21. Do national scholarships cover the full cost of education?")

    

def generate_response(input_text):
    model = ChatOllama(model="llama3.2", base_url="http://localhost:11434/")

    response = model.invoke(input_text)

    return response.content

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if submit and text:
    with st.spinner("Generating response..."):
        response = generate_response(text)
        st.session_state['chat_history'].append({"user": text, "ollama": response})
        st.write(response)

st.write("## Chat History")
for chat in reversed(st.session_state['chat_history']):
    st.write(f"**You**: {chat['user']}")
    st.write(f"**Blue bean**: {chat['ollama']}")
    st.write("---")  
