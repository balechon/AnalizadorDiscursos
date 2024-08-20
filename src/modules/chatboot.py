import os
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate


DB_PATH = db_path = os.path.join(os.path.dirname(__file__), "./../../data")
class RAGChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt_template = PromptTemplate(template="""
        Eres un asistente virtual que ayuda a responder preguntas sobre un discurso.
        
        Usa informacion general UNICAMENTE si esta en linea con la tematica y los puntos que aborda el discurso y no en otro escenario.
        
        Si la pregunta no está relacionada con el discurso o no tienes informacion sobre la tematica, debes responder: "Lo siento, no puedo responder a esa pregunta ya que no está relacionada con el discurso o no tengo información suficiente en el contexto proporcionado."
    
        Contexto: {context}
                        
        Pregunta del humano: {question} """,
           input_variables=["context", "chat_history", "question"]
        )
        self.vectorstore = None
        self.qa_chain = None

    def cargar_discurso(self, discurso):
        textos = self.text_splitter.split_text(discurso)

        self.vectorstore = Chroma.from_texts(collection_name='DISCURSO_DB',texts=textos, embedding=self.embeddings,persist_directory=DB_PATH)

        retriever = self.vectorstore.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            memory=self.memory,
            chain_type_kwargs={
                "prompt": self.prompt_template
            }
        )

    def responder_pregunta(self, pregunta):
        if not self.qa_chain:
            return "Por favor, carga primero el discurso."
        respuesta = self.qa_chain({"query": pregunta})
        return respuesta['result']

    def cerrar_sesion(self):
        self.vectorstore.delete_collection()
        self.vectorstore = None
        self.qa_chain = None
        self.memory.clear()


