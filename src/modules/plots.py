import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import plotly.express as px
from nltk.corpus import stopwords
import seaborn as sns


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
