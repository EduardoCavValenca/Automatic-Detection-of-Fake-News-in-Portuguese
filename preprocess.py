import os
import re
import pandas as pd

stopwords = ['de','a','o','que','e','do','da','em','um','para','e','com','nao','uma','os','no','se','na','por','mais','as',
            'dos','como','mas','foi','ao','ele','das','tem','a','seu','sua','ou','ser','quando','muito','ha','nos','ja',
            'esta','eu','tambem','so','pelo','pela','ate','isso','ela','entre','era','depois','sem','mesmo','aos','ter',
            'seus','quem','nas','me','esse','eles','estao','voce','tinha','foram','essa','num','nem','suas','meu','as',
            'minha','tem','numa','pelos','elas','havia','seja','qual','sera','nos','tenho','lhe','deles','essas','esses',
            'pelas','este','fosse','dele','tu','te','voces','vos','lhes','meus','minhas','teu','tua','teus','tuas','nosso',
            'nossa','nossos','nossas','dela','delas','esta','estes','estas','aquele','aquela','aqueles','aquelas','isto',
            'aquilo','estou','esta','estamos','estao','estive','esteve','estivemos','estiveram','estava','estavamos',
            'estavam','estivera','estiveramos','esteja','estejamos','estejam','estivesse','estivessemos','estivessem',
            'estiver','estivermos','estiverem','hei','ha','havemos','hao','houve','houvemos','houveram','houvera','houveramos',
            'haja','hajamos','hajam','houvesse','houvessemos','houvessem','houver','houvermos','houverem','houverei','houvera',
            'houveremos','houverao','houveria','houveriamos','houveriam','sou','somos','sao','era','eramos','eram','fui','foi',
            'fomos','foram','fora','foramos','seja','sejamos','sejam','fosse','fossemos','fossem','for','formos','forem','serei',
            'sera','seremos','serao','seria','seriamos','seriam','tenho','tem','temos','tem','tinha','tinhamos','tinham','tive',
            'teve','tivemos','tiveram','tivera','tiveramos','tenha','tenhamos','tenham','tivesse','tivessemos','tivessem','tiver',
            'tivermos','tiverem','terei','tera','teremos','terao','teria','teriamos','teriam']

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
    # remove stopwords  
    text = ' '.join([word for word in text.split() if word not in stopwords])
    # remove extra spaces 
    text = re.sub(r' +', ' ', text)

    return text


def processText(text):
    text = removeSpecialChars(text)
    text = lowerCase(text)
    text = removeStopWords(text)

    return text

# Read every file in the size_normalized and preprocess it to pre_normalized
def readCreateFiles():
    for folder in ['fake', 'true']:
        for file in os.listdir('size_normalized_texts/' + folder):
            with open('size_normalized_texts/' + folder + '/' + file, 'r', encoding='utf-8') as f:
                text = f.read()
                text = processText(text)
                with open('pre_normalized/' + folder + '/' + file, 'w', encoding='utf-8') as f2:
                    f2.write(text)

#Create new csv file with the pre_normalized text
def create_csv():
    newsDict = {}
    for folder in ['fake', 'true']:
        newsDict[folder] = {}
        for file in os.listdir('pre_normalized/' + folder):
            with open('pre_normalized/' + folder + '/' + file, 'r', encoding='utf-8') as f:
                text = f.read()
                newsDict[folder][file[:-4]] = (text, folder)

    # Read the files and create a dataframe for each folder
    true_df = pd.DataFrame.from_dict(newsDict["true"], orient='index', columns=['text', 'label'])
    fake_df = pd.DataFrame.from_dict(newsDict["fake"], orient='index', columns=['text', 'label'])

    # Join both dataframes
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Sort the dataframe by label
    df = df.sort_values(by=['label']).reset_index(drop=True)

    # Save the dataframe to a csv file
    df.to_csv('news.csv', index=False)


def main():
    readCreateFiles()       

if __name__ == "__main__":
    main()
   
    


    
