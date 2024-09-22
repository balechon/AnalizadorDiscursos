import os
import logging
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.llm = ChatOpenAI(temperature=0, model=os.getenv("MODEL_OPENAI_ID"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt_template = PromptTemplate(
            template="""
Eres un fact-checker que está revisando un discurso político. Usa la información proporcionada y tu conocimiento general para responder la pregunta del usuario.

Contexto adicional relevante:
{context}

Pregunta: 
{question}

El discurso completo es el siguiente:
- **Discurso:** {speech}

- **Historial de la conversación:** {chat_history}
            """,
            input_variables=["context", "speech", "chat_history", "question"]
        )
        self.speech = None

        try:
            # Inicializar Pinecone
            pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
            self.index_name = os.environ.get("INDEX_NAME")

            # Inicializar el vectorstore con el índice existente
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.vectorstore = Pinecone.from_existing_index(self.index_name, embeddings)
            logger.info(f"Pinecone index '{self.index_name}' connected successfully.")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    def cargar_discurso(self, discurso, sobrescribir=False):
        if self.speech and not sobrescribir:
            return "Ya existe un discurso cargado. Utiliza sobrescribir=True para reemplazarlo."
        self.speech = discurso
        return "Discurso cargado exitosamente."

    def responder_pregunta(self, pregunta):
        if not self.speech:
            return "Por favor, carga primero el discurso."

        chat_history = self.memory.load_memory_variables({})["chat_history"]

        try:
            # Realizar la búsqueda de contexto relevante en Pinecone
            docs = self.vectorstore.similarity_search(pregunta, k=10)

            logger.info(f"Retrieved {len(docs)} documents from Pinecone")

            # Imprimir los documentos extraídos para depuración
            print("\n--- Documentos extraídos de Pinecone ---")
            for i, doc in enumerate(docs, 1):
                print(f"Documento {i}:")
                if hasattr(doc, 'page_content'):
                    print(f"Contenido: {doc.page_content[:200]}...")  # Primeros 200 caracteres
                elif 'text' in doc.metadata:
                    print(f"Contenido (de metadata): {doc.metadata['text'][:200]}...")
                else:
                    print("Documento no tiene contenido textual")
                print(f"Metadatos: {doc.metadata if hasattr(doc, 'metadata') else 'No metadata'}")
                print("---")

            # Extraer el contenido de texto de los documentos
            context = "\n".join([
                doc.page_content if hasattr(doc, 'page_content')
                else doc.metadata.get('text', '') if hasattr(doc, 'metadata')
                else ''
                for doc in docs
            ])

            if not context.strip():
                logger.warning("No se pudo extraer contenido de los documentos")
                context = "No se pudo recuperar contexto relevante."

        except Exception as e:
            logger.error(f"Error retrieving documents from Pinecone: {e}")
            context = "Error retrieving context."

        prompt = self.prompt_template.format(
            context=context,
            speech=self.speech,
            chat_history=chat_history,
            question=pregunta
        )

        respuesta = self.llm.predict(prompt)

        # Actualizar la memoria con la pregunta y respuesta
        self.memory.save_context({"input": pregunta}, {"output": respuesta})

        return respuesta

    def cerrar_sesion(self):
        self.speech = None
        self.memory.clear()