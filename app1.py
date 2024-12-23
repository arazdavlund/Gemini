# Importera bibliotek
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import chromadb.api
from dotenv import load_dotenv
from langdetect import detect
import os

# Rensa cache
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Ladda miljövariabler
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Streamlit inställningar(log)
st.image("sigtuna.png", width=311)
st.markdown("<h2 style='text-align: center;'>Välkommen till Sigtuna kommun Chattbot</h2>", unsafe_allow_html=True)

# Ladda flera PDF-filer
pdf_files = ["rutin_avvikelser.pdf", "rutin_upphandlingsreglemente.pdf"]
all_data = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    data = loader.load()
    all_data.extend(data)  # Lägg till data från varje PDF

# Dela upp dokument i mindre delar
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(all_data)

# Skapa en vektorbutik för dokument
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Skapa en retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# LLM-inställningar
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Streamlit input och logik
query = st.chat_input("Skicka ett meddelande till Sigtuna kommun Chattbot: ") 
if query:
    # Visa användarens fråga
    with st.chat_message("user"):
        st.write(query)
    
    # Visa "Assistant" meddelande
    with st.chat_message("assistant"):
        # Identifiera språk
        language = detect(query)
        
        if language == "sv":  # Om frågan är på svenska
            system_prompt = (
                "Du är en assistent för att svara på frågor. "
                "Använd följande delar av kontexten för att svara "
                "på frågan. Om du inte vet svaret, säg att du "
                "inte vet. Använd högst tre meningar och håll "
                "svaret kortfattat. Alla svar ska vara på svenska."
                "\n\n"
                "{context}"
            )
        else:  # Om frågan är på engelska eller annat språk
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Skapa RAG-kedja
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Generera svar
        response = rag_chain.invoke({"input": query})

        # Kontrollera om svar hittades
        if response and "answer" in response and response["answer"].strip():
            # Om svaret är på engelska, översätt det till svenska
            if detect(response["answer"]) != "sv":
                translation_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Du är en översättare. Översätt följande text till svenska och behåll samma ton och stil."),
                    ("human", response["answer"])
                ])
                translation_chain = translation_prompt | llm
                translated_response = translation_chain.invoke({})
                st.write(translated_response.content)
            else:
                st.write(response["answer"])
        else:
            st.write("Jag kunde tyvärr inte hitta något relevant svar. Försök att omformulera din fråga.")