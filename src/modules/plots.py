import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import plotly.express as px
from nltk.corpus import stopwords
import seaborn as sns
import numpy as np

def plot_top_words(text, top_n=20):
    words = text.split()
    # remove stopwords
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word.lower() not in stop_words]
    words = [word for word in words if len(word) > 3]
    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)
    df = pd.DataFrame(top_words, columns=['word', 'count'])
    df = df.sort_values('count', ascending=False)
    total = df['count'].sum()
    df['percentage'] = (df['count'] / total) * 100

    pal = list(sns.color_palette(palette='BuGn_r', n_colors=top_n+1).as_hex())
    fig = go.Figure(data=[go.Pie(
        labels=df['word'],
        values=df['percentage'],
        hole=.6,
        marker_colors=pal[:top_n],
        textposition='outside',
        textinfo='label+percent',
        hoverinfo='label+percent',
        textfont_size=10,
        insidetextorientation='radial'
    )])

    fig.update_layout(
        # title_text=f"Top {top_n} palabras más frecuentes",
        showlegend=True,
        # annotations=[dict(text='Palabras', x=0.5, y=0.5, font_size=20, showarrow=False)],
        width=1024, height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def plot_overal_sentiment_score(sentiment:dict):
    promedios = {
        sentimiento: np.mean(valores) if valores else 0
        for sentimiento, valores in sentiment.items()
    }

    # Crear el gráfico de dona
    fig = go.Figure(data=[go.Pie(
        labels=list(promedios.keys()),
        values=list(promedios.values()),
        hole=.5,
        marker_colors=['#00cc96', '#ffa15a', '#ef553b']  # Verde, Naranja, Rojo
    )])

    # Personalizar el diseño
    fig.update_layout(
        title_text="Análisis General del Texto",
        showlegend=True,
        legend_title="Sentimientos",
        height=500,
        width=700
    )

    return fig


def plot_sentiment_stacked_bar(sentiment: dict):
    # Preparar los datos
    ideas = list(range(1, len(sentiment['positive']) + 1))
    positive = sentiment['positive']
    neutral = sentiment['neutral']
    negative = sentiment['negative']

    # Crear el gráfico de barras apiladas
    fig = go.Figure(data=[
        go.Bar(name='Positivo', x=ideas, y=positive, marker_color='#00cc96'),
        go.Bar(name='Neutral', x=ideas, y=neutral, marker_color='#ffa15a'),
        go.Bar(name='Negativo', x=ideas, y=negative, marker_color='#ef553b')
    ])

    # Cambiar el modo de apilamiento
    fig.update_layout(barmode='stack')

    # Personalizar el diseño
    fig.update_layout(
        title_text="Análisis de Sentimiento por Idea",
        xaxis_title="Idea",
        yaxis_title="Puntuación de Sentimiento",
        legend_title="Sentimientos",
        height=500,
        width=700
    )

    return fig


def plot_sentiment_stacked_area(sentiment: dict):
    # Preparar los datos
    ideas = list(range(1, len(sentiment['positive']) + 1))

    # Crear el gráfico de áreas apiladas
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ideas, y=sentiment['positive'],
        mode='lines', name='Positivo',
        line=dict(width=0.5, color='#00cc96'),
        stackgroup='one', fillcolor='#00cc96'
    ))
    fig.add_trace(go.Scatter(
        x=ideas, y=sentiment['neutral'],
        mode='lines', name='Neutral',
        line=dict(width=0.5, color='#ffa15a'),
        stackgroup='one', fillcolor='#ffa15a'
    ))
    fig.add_trace(go.Scatter(
        x=ideas, y=sentiment['negative'],
        mode='lines', name='Negativo',
        line=dict(width=0.5, color='#ef553b'),
        stackgroup='one', fillcolor='#ef553b'
    ))

    # Personalizar el diseño
    fig.update_layout(
        title='Evolución del Sentimiento por Idea (Áreas Apiladas)',
        xaxis_title='Idea / Fragmento de Tiempo',
        yaxis_title='Proporción de Sentimiento',
        legend_title='Sentimientos',
        height=500,
        width=700,
        template='plotly_dark',  # Para mantener la consistencia con tus otros gráficos
        yaxis=dict(range=[0, 1])  # Asegura que el eje Y va de 0 a 1 para mostrar proporciones
    )

    return fig