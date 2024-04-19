import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from modules.gpt import GPT
from modules.gpt import WhisperModel
from tempfile import  NamedTemporaryFile
from io import BytesIO


st.set_page_config(
    page_title="Analizador Discursos",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

def app():

    gpt = GPT()
    whisper = WhisperModel()

    st.title('Analizador de Discursos')

    st.subheader('Sube un audio en Español y obtén un resumen del discurso')

    format_list = ['mp3', 'mp4', 'wav', 'm4a']
    format_string = ', '.join(format_list)
    audio_file = st.file_uploader(f"Sube tu Audio en formato {format_string}", type=format_list)
    btn = st.button('Cargar Audio')

    if btn:

        if audio_file is None:
            st.warning('Sube un archivo valido')
            return

        status = st.status('Creando la Transcripcion', expanded=False, state='running')

        # Llamada a la función de transcripción
        try:
            resultado_transcripcion = whisper.transcribe_audio(audio_file)
            st.text_area("Texto transcrito", resultado_transcripcion, height=300)
        except Exception as e:
            st.error(f"Error en la transcripción: {e}")


if __name__ == '__main__':
	app()