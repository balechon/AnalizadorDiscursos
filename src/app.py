import streamlit as st
from dotenv import load_dotenv
from modules.whisper_transcriptor import WhisperModel
from modules.gpt import GPT
load_dotenv()
from modules.sentiment_analyzer import text_classifier, extract_sentiment_metrics
st.set_page_config(page_title="Analizador de Discursos IA", layout="wide")

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización de modelos
@st.cache_resource
def load_models():
    return WhisperModel(), GPT()


whisper_model, gpt_model = load_models()


# Funciones de análisis
@st.cache_data
def transcribe_audio(audio_file):
    return whisper_model.transcribe_audio(audio_file)


@st.cache_data
def generate_summary(text):
    return gpt_model.create_resume(text=text)

@st.cache_data
def text_segmentation(text):
    return gpt_model.divide_into_ideas(text)

@st.cache_data
def perform_sentiment_analysis(text):
    list_sentimen_analisis = {"positive": [], "neutral": [], "negative": []}
    ideas = text_segmentation(text)
    for idea in ideas:
        sentiment_scores = text_classifier(idea)
        for key_sentiment in list_sentimen_analisis.keys():
            list_sentimen_analisis[key_sentiment].append(sentiment_scores[key_sentiment])
    return extract_sentiment_metrics(list_sentimen_analisis)


# Función para el chatbot (simplificada)
def chatbot_response(question, context):
    # Aquí implementarías la lógica del chatbot
    return f"Respuesta a: {question}\nBasada en el contexto del discurso analizado."

st.markdown("""
    <style>
    .reportview-container .main .block-container{
        max-width: 1000px;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Interfaz principal
def main():
    # st.title("Analizador de Discursos IA",)
    st.markdown(
        "<h1 style='text-align: center;'>Analizador de Discursos IA</h1>",
        unsafe_allow_html=True)
    # Inicialización de estado de sesión
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'text' not in st.session_state:
        st.session_state.text = ''
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None

    # Página de inicio
    if st.session_state.page == 'home':
        st.markdown(
            "<h3 style='text-align: center;'>Analiza, resume y comprende discursos con inteligencia artificial</h3>",
            unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Subir audio"):
                st.session_state.page = 'upload_audio'
                st.experimental_rerun()
        with col2:
            if st.button("Ingresar texto"):
                st.session_state.page = 'enter_text'
                st.experimental_rerun()

    # Página de subir audio
    elif st.session_state.page == 'upload_audio':
        st.subheader("Sube tu archivo de audio")
        audio_file = st.file_uploader("Elige un archivo de audio", type=['mp3', 'wav', 'm4a'])
        if audio_file is not None:
            st.audio(audio_file)
            if st.button("Transcribir y Analizar"):
                with st.spinner("Transcribiendo audio..."):
                    st.session_state.text = transcribe_audio(audio_file)
                st.session_state.page = 'analysis'
                st.experimental_rerun()
        if st.button("Volver al Inicio"):
            st.session_state.page = 'home'
            st.session_state.text = ''
            st.session_state.analysis = None
            st.experimental_rerun()
    # Página de ingresar texto
    elif st.session_state.page == 'enter_text':
        st.subheader("Ingresa el texto de tu discurso")
        st.session_state.text = st.text_area("Texto del discurso", height=300)
        if st.button("Analizar"):
            st.session_state.page = 'analysis'
            st.experimental_rerun()
        if st.button("Volver al Inicio"):
            st.session_state.page = 'home'
            st.session_state.text = ''
            st.session_state.analysis = None
            st.experimental_rerun()
    # Página de análisis
    elif st.session_state.page == 'analysis':
        st.sidebar.title("Secciones de Análisis")
        analysis_section = st.sidebar.radio("",
                                            ['Resumen', 'Análisis de Sentimientos', 'Palabras Clave', 'Estadísticas',
                                             'Chatbot IA'])

        st.subheader("Texto Original")
        st.text_area("", st.session_state.text, height=150)

        if analysis_section == 'Resumen':
            if 'summary' not in st.session_state:
                with st.spinner("Generando resumen..."):
                    st.session_state.summary = generate_summary(st.session_state.text)
            st.subheader("Resumen del Discurso")
            st.write(st.session_state.summary['summary'])


        elif analysis_section == 'Análisis de Sentimientos':
            st.subheader("Análisis de Sentimientos")
            if 'sentiment_analysis' not in st.session_state:
                with st.spinner("Realizando análisis de sentimientos..."):
                    st.session_state.sentiment_analysis = perform_sentiment_analysis(st.session_state.text)

            st.write(st.session_state.sentiment_analysis)

        elif analysis_section == 'Chatbot IA':
            st.subheader("Chatbot IA")
            user_question = st.text_input("Haz una pregunta sobre el discurso:")
            if user_question:
                response = chatbot_response(user_question, st.session_state.text)
                st.write(response)


        if st.sidebar.button("Volver al Inicio"):
            st.session_state.page = 'home'
            st.session_state.text = ''
            st.session_state.analysis = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()