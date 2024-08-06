from transformers import pipeline
import pandas as pd

MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# Inicializar el pipeline de clasificación de texto
pipe = pipeline("text-classification",
                model=MODEL_NAME,
                top_k=None
                )

def text_classifier(texto):
    results = pipe(texto)[0]
    sentiments_score = { result['label']: result['score'] for result in  results }
    return sentiments_score


def extract_sentiment_metrics(metrics):
    df_metrics = pd.DataFrame(metrics)
    sentences_analyzed = len(df_metrics)
    positive_average = df_metrics['positive'].mean()
    neutral_average = df_metrics['neutral'].mean()
    negative_average = df_metrics['negative'].mean()
    string_response = f"Average sentiment scores:\nPositive: {positive_average}\nNeutral: {neutral_average}\nNegative: {negative_average}"
    return string_response




