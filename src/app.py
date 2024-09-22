import streamlit as st
from dotenv import load_dotenv
from modules.whisper_transcriptor import WhisperModel
from modules.gpt import GPT
from modules.chatboot import RAGChatbot
from modules.sentiment_analyzer import text_classifier, extract_sentiment_metrics
from modules.NLP_basics import normalize_text
from modules.plots import plot_top_words,plot_overal_sentiment_score,plot_sentiment_stacked_bar,plot_sentiment_stacked_area


load_dotenv()

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Analizador de Discursos IA", layout="wide")
# Inicialización de modelos
@st.cache_resource
def load_models():
    return WhisperModel(), GPT()


whisper_model, gpt_model  =  load_models()



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
    return extract_sentiment_metrics(list_sentimen_analisis), list_sentimen_analisis, ideas

@st.cache_data
def plot_words(text):
    return plot_top_words(text)




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
                st.rerun()
        with col2:
            if st.button("Ingresar texto"):
                st.session_state.page = 'enter_text'
                st.rerun()

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
                st.rerun()
        if st.button("Volver al Inicio", key="main_home"):
            st.session_state.page = 'home'
            st.session_state.text = ''
            st.session_state.analysis = None
            st.rerun()


    # Página de ingresar texto
    elif st.session_state.page == 'enter_text':
        st.subheader("Ingresa el texto de tu discurso")
        st.session_state.text = st.text_area("Texto del discurso", height=300)
        if st.button("Analizar"):
            st.session_state.page = 'analysis'
            st.rerun()
        if st.button("Volver al Inicio", key="main_home"):
            st.session_state.page = 'home'
            st.session_state.text = ''
            st.session_state.analysis = None
            st.rerun()
    # Página de análisis
    elif st.session_state.page == 'analysis':
        st.sidebar.title("Secciones de Análisis")
        analysis_section = st.sidebar.radio("",
                                            ['Resumen', 'Análisis de Sentimientos', 'Chatbot IA'])

        st.subheader("Texto Original")
        with st.expander("Expandir todo el texto"):
            st.write(st.session_state.text)
        # st.text_area("", st.session_state.text, height=150)

        if analysis_section == 'Resumen':
            if 'summary' not in st.session_state:
                with st.spinner("Generando resumen..."):
                    st.session_state.summary = generate_summary(st.session_state.text)
            st.subheader("Resumen del Discurso")
            st.write(st.session_state.summary['summary'])

            normalized_text = normalize_text(st.session_state.text)
            st.subheader("Palabras más frecuentes")
            fig = plot_words(normalized_text)
            st.plotly_chart(fig)


        elif analysis_section == 'Análisis de Sentimientos':
            st.subheader("Análisis de Sentimientos")
            if 'sentiment_analysis' not in st.session_state:
                with st.spinner("Realizando análisis de sentimientos..."):
                    string_sentiment_result,list_sentiment_result, ideas = perform_sentiment_analysis(st.session_state.text)
                    st.session_state.sentiment_analysis = string_sentiment_result
                    st.session_state.ideas_sentiment_result = list_sentiment_result
                    st.session_state.ideas = ideas
            st.subheader("Texto dividido en Ideas")
            with st.expander("Ver todas las ideas"):
                for idea in st.session_state.ideas:
                    st.write(idea)

            overall_sentiment_fig = plot_overal_sentiment_score(st.session_state.ideas_sentiment_result)
            ideas_sentiment_fig = plot_sentiment_stacked_bar(st.session_state.ideas_sentiment_result)
            ideas_staked_fig = plot_sentiment_stacked_area(st.session_state.ideas_sentiment_result)

            st.plotly_chart(overall_sentiment_fig)
            st.plotly_chart(ideas_sentiment_fig)
            st.plotly_chart(ideas_staked_fig)
            # st.write(st.session_state.sentiment_analysis)

        elif analysis_section == 'Chatbot IA':
            st.subheader("Chatbot IA")

            # Inicializar el chatbot si no existe en el estado de sesión
            if 'chatbot' not in st.session_state:
                st.session_state.chatbot = RAGChatbot()

            # Cargar el discurso en el chatbot si aún no se ha hecho
            if 'chatbot_loaded' not in st.session_state or not st.session_state.chatbot_loaded:
                with st.spinner("Cargando el discurso en el chatbot..."):
                    st.session_state.chatbot.cargar_discurso(st.session_state.text)
                    st.session_state.chatbot_loaded = True

            # Inicializar el historial de mensajes si no existe
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            # Mostrar el historial de mensajes
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Campo de entrada para la pregunta del usuario
            user_question = st.chat_input("Haz una pregunta sobre el discurso:")

            if user_question:
                # Añadir la pregunta del usuario al historial
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                # Generar y mostrar la respuesta del chatbot
                with st.chat_message("assistant"):
                    with st.spinner("Generando respuesta..."):
                        try:
                            # Obtener la respuesta del chatbot
                            response = st.session_state.chatbot.responder_pregunta(user_question)

                            # Asegurarse de que la respuesta es una cadena
                            if not isinstance(response, str):
                                response = str(response)

                            # Mostrar la respuesta
                            st.markdown(response)

                            # Añadir la respuesta del chatbot al historial
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error al generar la respuesta: {str(e)}")

            # Botón para reiniciar la conversación
            if st.button("Reiniciar conversación"):
                st.session_state.messages = []
                st.session_state.chatbot.cerrar_sesion()  # Limpiar la sesión del chatbot
                st.session_state.chatbot_loaded = False
                st.experimental_rerun()

        if st.sidebar.button("Volver al Inicio", key="sidebar_home"):
            st.session_state.page = 'home'
            st.session_state.text = ''
            st.session_state.analysis = None
            # clear cache
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.cerrar_sesion()
            st.session_state.chatbot_loaded = False

            st.cache_data.clear()

            for key in ['summary', 'sentiment_analysis', 'chatbot_loaded','chatbot','messages']:
                if key in st.session_state:
                    del st.session_state[key]


            st.rerun()

if __name__ == "__main__":
    main()