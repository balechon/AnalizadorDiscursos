import json
import hashlib
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone
from tqdm.auto import tqdm
import os
from src.modules.paths_reference import ROOT_PATH

CACHE_FILE = ROOT_PATH / "data" / "document_cache.json"


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def ingest_docs(documents, embeddings, batch_size=100):
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = os.environ.get("INDEX_NAME")
        index = pc.Index(index_name)
    except Exception as e:
        print(f"Error al inicializar Pinecone: {e}")
        return

    print("\nCargando documentos en Pinecone...")

    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i + batch_size]
        ids = [f"doc_{j}" for j in range(i, i + len(batch))]
        texts = [doc.page_content for doc in batch]
        metadatas = [
            {**doc.metadata, 'text': doc.page_content}  # Incluir el texto dentro de la metadata
            for doc in batch
        ]

        try:
            embeddings_list = embeddings.embed_documents(texts)
            to_upsert = list(zip(ids, embeddings_list, metadatas))
            index.upsert(vectors=to_upsert)
        except Exception as e:
            print(f"Error al procesar el lote {i // batch_size}: {e}")

    print("Carga completa")


def run():
    load_dotenv()

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    pdf_folder = ROOT_PATH / "data" / "pdfs"
    pdf_files = [pdf_folder / f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    cache = load_cache()
    all_documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    for pdf_file in tqdm(pdf_files, desc="Procesando PDFs"):
        file_hash = get_file_hash(pdf_file)
        if file_hash in cache:
            print(f"Saltando archivo ya procesado: {pdf_file.name}")
            continue

        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()
            for page_num, doc in enumerate(pdf_docs):
                chunks = text_splitter.split_text(doc.page_content)
                for chunk_num, chunk in enumerate(chunks):
                    document = Document(
                        page_content=chunk,
                        metadata={
                            'source': str(pdf_file.name),
                            'page': page_num + 1,
                            'chunk': chunk_num + 1,
                            'text': chunk  # Añadir el texto aquí
                        }
                    )
                    all_documents.append(document)

            # Añadir el archivo procesado al caché
            cache[file_hash] = pdf_file.name
        except Exception as e:
            print(f"Error al procesar {pdf_file}: {e}")

    if all_documents:
        ingest_docs(all_documents, embeddings)

    # Guardar el caché actualizado
    save_cache(cache)


if __name__ == "__main__":
    run()