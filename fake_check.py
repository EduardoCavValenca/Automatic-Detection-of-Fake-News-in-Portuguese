import requests
import pandas as pd
from collections import defaultdict

def pad_article(article):
    original_article = article
    while len(article.split()) < 100:
        article += "\n\n" + original_article
    return article

def generate_response(article:str, model:str):
    response = requests.post('http://nilc-fakenews.herokuapp.com/ajax/check_web/', 
                        data={'text': article, 'model': model}, 
                        cookies={'csrftoken': 'xr3HZGxMUGv5GL8CZ2e2ZhOZXew7iaHPzgdeEhmVDupCyHeh5gD6EaHyrhy2A6aZ'}, 
                        headers={
                            'Origin': 'https://nilc-fakenews.herokuapp.com/',
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
                            'Referer': 'https://nilc-fakenews.herokuapp.com/',
                            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                            'Host': 'nilc-fakenews.herokuapp.com',
                            'X-CSRFToken': 'xr3HZGxMUGv5GL8CZ2e2ZhOZXew7iaHPzgdeEhmVDupCyHeh5gD6EaHyrhy2A6aZ',
                            'X-Requested-With': 'XMLHttpRequest'})
    return response

def fake_check_response(data : pd.DataFrame, padding : bool = True):

    convert_label = {'REAL': 'true', 'FAKE': 'fake'}
    
    models = ['unigramas', 'pos']
    models_name = ['Palavras do texto', 'Classes gramaticais']

    return_data = defaultdict(list)

    for model, model_name in zip(models, models_name):
        for article, label in zip(data['text'], data['label']):
            if padding:
                article = pad_article(article)

            response = generate_response(article, model)
            pred = convert_label[response.json()['result']]

            return_data['text'].append(article)
            return_data['label'].append(label)
            return_data['model'].append(model_name)
            return_data['prediction'].append(pred)

    return pd.DataFrame(return_data)       
    