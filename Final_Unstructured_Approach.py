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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None return
    print("****** Extracted Text *******",text)
    return text

def preprocess_text(text):
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(lemmas)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def preprocess_question(question):
    # Simple preprocessing to normalize question phrasing
    # This is a placeholder for more sophisticated preprocessing if needed
    question = question.lower().strip()
    return question

def update_vector_store(text_chunks, vector_store=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if vector_store is None:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    else:
        # Add new text chunks to the existing vector store
        vector_store.add_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
# Alternate Prompt
# systemprompt= Act as an expert chatbot AI, provide accurate, detailed, and precise answers, ensuring completeness without unnecessary verbosity. Directly address the question's core, strictly adhering to the provided context. If information is lacking, respond with 'The answer is not available in the context'. Your responses must be relevant, clear, and adhere to these guidelines in every scenario very strictly.
def get_conversational_chain():
    prompt_template = """
    "Act as an expert chatbot AI, provide accurate, detailed, and precise answers, ensuring completeness without unnecessary verbosity. Directly address the question's core, strictly adhering to the provided context. If information is lacking, respond with 'The answer is not available in the context'. Your responses must be relevant, clear, and adhere to these guidelines in every scenario very strictly."

    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user questions
def user_input(user_question):
    user_question = preprocess_question(user_question)  # Preprocess question
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the vector store from the session state
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=3)  # Retrieve top 3 similar documents

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with UnStructured Files using Gemini pro")

    # Initialize session state for uploaded files and processed text
    if 'processed_text' not in st.session_state:
        st.session_state['processed_text'] = ""
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None

    # Sidebar - Stable UI
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process each uploaded file and append its content to the session state
                for pdf in pdf_docs:
                    raw_text = get_pdf_text([pdf])
                    processed_text = preprocess_text(raw_text)
                    st.session_state['processed_text'] += " " + processed_text

                # Split the updated processed text into chunks and update the vector store
                text_chunks = get_text_chunks(st.session_state['processed_text'])
                st.session_state['vector_store'] = update_vector_store(text_chunks, st.session_state['vector_store'])
                st.success("Done")

    # Main Content Area
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question and st.button("Get Answer"):
        user_input(user_question)

if __name__ == "__main__":
    main()