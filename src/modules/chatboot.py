import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter


class RAGChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.vectorstore = None
        self.qa_chain = None

    def cargar_discurso(self, discurso):
        textos = self.text_splitter.split_text(discurso)
        self.vectorstore = Chroma.from_texts(texts=textos, embedding=self.embeddings)
        retriever = self.vectorstore.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )

    def responder_pregunta(self, pregunta):
        if not self.qa_chain:
            return "Por favor, carga primero el discurso."
        respuesta = self.qa_chain.run(pregunta)
        return respuesta

    def cerrar_sesion(self):
        if self.vectorstore:
            self.vectorstore = None
        self.qa_chain = None
        self.memory.clear()




