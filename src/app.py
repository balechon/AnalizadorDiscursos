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
    layout="wide",
    initial_sidebar_state="expanded",
)


def app():
    gpt = GPT()
    whisper = WhisperModel()

    st.container().title('Analizador de Discursos')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('Sube un audio y obtén un Análisis del discurso')

        format_list = ['mp3', 'mp4', 'wav', 'm4a']
        format_string = ', '.join(format_list)
        audio_file = st.file_uploader(f"Sube tu Audio en formato {format_string}", type=format_list)


        if audio_file is not None:
            # Mostrar reproductor de audio
            st.audio(audio_file)

            if st.button('Obtener Transcripcion'):


                with col2:
                    status = st.status('Creando la Transcripción', expanded=False, state='running')

                    # Llamada a la función de transcripción
                    try:
                        result_text_whisper = whisper.transcribe_audio(audio_file)
                        status.update(label='Texto transcrito', state='complete')
                        st.session_state['texto_transcrito'] = result_text_whisper
                        editable_text = st.text_area("Texto Transcrito:", value=result_text_whisper, height=300, key="editable")

                        if st.button("Guardar Cambios"):
                            st.session_state['texto_transcrito'] = editable_text

                        if st.button("Generar Resumen"):
                            try:
                                summary = gpt.summarize_text(st.session_state['texto_transcrito'])
                                st.text_area("Resumen del Discurso:", value=summary, height=150)
                            except Exception as e:
                                st.error(f"Error al generar resumen: {e}")

                    except Exception as e:
                        st.error(f"Error en la transcripción: {e}")

if __name__ == '__main__':
	app()