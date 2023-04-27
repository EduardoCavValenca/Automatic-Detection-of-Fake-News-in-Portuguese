import os
import re
import pandas as pd

#Caso nao tenha instalado rode as seguinte duas linhas
# import nltk
# nltk.download('stopwords')

#Caso ja esteja instalado
from nltk.corpus import stopwords as nltk_stopwords

def removeSpecialChars(text):
    # Remove all non alphanumeric characters (keep spaces and letters with accents)
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)

    # Remove accents
    text = re.sub(r'À|Á|Â|Ã|Ä|Å|à|á|â|ã|ä|å', 'a', text)
    text = re.sub(r'È|É|Ê|Ë|è|é|ê|ë', 'e', text)
    text = re.sub(r'Ì|Í|Î|Ï|ì|í|î|ï', 'i', text)
    text = re.sub(r'Ò|Ó|Ô|Õ|Ö|Ø|ò|ó|ô|õ|ö|ø', 'o', text)
    text = re.sub(r'Ù|Ú|Û|Ü|ù|ú|û|ü', 'u', text)
    text = re.sub(r'Ý|ý|ÿ', 'y', text)
    text = re.sub(r'Ç|ç', 'c', text)
    text = re.sub(r'Ñ|ñ', 'n', text)

    return text

def lowerCase(text):
    # make everything lowercase
    text = text.lower()

    return text

def removeStopWords(text):

    stopwords = nltk_stopwords.words('portuguese')
    #Nao inclui nessa, nesta ... etc

    # remove stopwords  
    text = ' '.join([word for word in text.split() if word not in stopwords])
    # remove extra spaces 
    text = re.sub(r' +', ' ', text)
    text = re.sub('[\n]+', ' ', text)
    text = re.sub(r'[\r]+', ' ', text)
    text = re.sub(r'[\t]+', ' ', text)

    return text


def processText(text):
    text = removeSpecialChars(text)
    text = lowerCase(text)
    text = removeStopWords(text)

    return text

# Read every file in the size_normalized and preprocess it to pre_normalized
def readCreateFiles():
    for folder in ['fake', 'true']:
        for file in os.listdir('./data/text_files/size_normalized_texts/' + folder):
            with open('./data/text_files/size_normalized_texts/' + folder + '/' + file, 'r', encoding='utf-8') as f:
                text = f.read()
                text = processText(text)
                with open('./data/text_files/pre_normalized_with_stopwords/' + folder + '/' + file, 'w', encoding='utf-8') as f2:
                    f2.write(text)

#Create new csv file with the pre_normalized text
def create_csv():
    newsDict = {}
    for folder in ['fake', 'true']:
        newsDict[folder] = {}
        for file in os.listdir('./data/text_files/pre_normalized/' + folder):
            with open('./data/text_files/pre_normalized/' + folder + '/' + file, 'r', encoding='utf-8') as f:
                text = f.read()
                newsDict[folder][file[:-4]] = (processText(text), folder)

    # Read the files and create a dataframe for each folder
    true_df = pd.DataFrame.from_dict(newsDict["true"], orient='index', columns=['text', 'label'])
    fake_df = pd.DataFrame.from_dict(newsDict["fake"], orient='index', columns=['text', 'label'])

    # Join both dataframes
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Sort the dataframe by label
    df = df.sort_values(by=['label']).reset_index(drop=True)

    # Save the dataframe to a csv file
    df.to_csv('./data/csvs/pre_normalized_with_stopwords.csv', index=False)


def main():
    create_csv()       

if __name__ == "__main__":
    main()
   
    


    
