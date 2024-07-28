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




# if __name__ == '__main__':
#
#     lista_prueba=[{'positive': 0.9530580043792725, 'neutral': 0.025432055816054344, 'negative': 0.021509924903512},{'positive': 0.9530580043792725, 'neutral': 0.025432055816054344, 'negative': 0.021509924903512},{'positive': 0.9530580043792725, 'neutral': 0.025432055816054344, 'negative': 0.021509924903512},{'positive': 0.9530580043792725, 'neutral': 0.025432055816054344, 'negative': 0.021509924903512}]
#     # iput_text = input("Ingresa un texto para clasificar: ")
#     # result = text_classifier(iput_text)
#     # print(result)
#     final_result = {"positive": [], "neutral": [], "negative": []}
#     for i in lista_prueba:
#         for key in i.keys():
#             final_result[key].append(i[key])
#     df = pd.DataFrame(final_result)
#     positive_average = df['positive'].mean()
#     print(df.head())
#     print(positive_average)