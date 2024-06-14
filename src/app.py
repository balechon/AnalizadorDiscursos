import streamlit as st
from dotenv import load_dotenv
from modules.gpt import GPT, WhisperModel

st.set_page_config(
    page_title="Analizador de Discursos",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def app():
    if 'texto_transcrito' not in st.session_state:
        st.session_state['texto_transcrito'] = ""

    gpt = GPT()
    whisper = WhisperModel()

    st.container().title('Analizador de Discursos')
    col1, col2 = st.columns([0.7, 1])

    with col1:
        st.subheader('Sube un audio y obtén un Análisis del discurso')
        format_list = ['mp3', 'mp4', 'wav', 'm4a']
        format_string = ', '.join(format_list)
        audio_file = st.file_uploader(f"Sube tu Audio en formato {format_string}", type=format_list)

        if audio_file is not None:
            st.audio(audio_file)
            if st.button('Obtener Transcripción'):
                try:
                    result_text_whisper = whisper.transcribe_audio(audio_file)
                    st.session_state['texto_transcrito'] = result_text_whisper
                except Exception as e:
                    st.error(f"Error en la transcripción: {e}")

        if st.session_state['texto_transcrito']:
            editable_text = st.text_area("Texto Transcrito:", value=st.session_state['texto_transcrito'], height=300,
                                         key="editable")
            if st.button("Guardar Cambios"):
                st.session_state['texto_transcrito'] = editable_text
    with col2:
            if st.button("Generar Resumen"):
                try:
                    promt = f'Genera un resumen del siguiente texto: {st.session_state["texto_transcrito"]}'
                    summary = gpt.create_response(promt)
                    st.text_area("Resumen del Discurso:", value=summary, height=150)
                except Exception as e:
                    st.error(f"Error al generar resumen: {e}")


if __name__ == '__main__':
    load_dotenv()
    app()
