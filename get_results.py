import pandas as pd
import numpy as np
from time import perf_counter_ns
from collections import defaultdict

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def get_result_data_frames(algs : list, algs_names: list, vectorizers: list, vectorizers_names: list, data_frame_paths: list, data_frame_names: list, news_validation_path: str):

    scores = defaultdict(list)
    
    news_validation_df = pd.read_csv(news_validation_path)
    for i,file_path in enumerate(data_frame_paths):
        df = pd.read_csv(file_path)
        get_result_vectorizers(algs, algs_names, vectorizers, vectorizers_names, df, data_frame_names[i], scores, news_validation_df)

    return pd.DataFrame(scores)
    
def get_result_vectorizers(algs : list, algs_names: list, vectorizers: list, vectorizers_names: list, df: pd.DataFrame, data_frame_name: str, scores: dict, news_validation_df: pd.DataFrame):
    for i, vectorizer in enumerate(vectorizers):
        get_result_algs(algs, algs_names, vectorizer, vectorizers_names[i], df, data_frame_name, scores, news_validation_df)
    

def get_result_algs(algs : list, algs_names: list, vectorizer, vectorizer_name: str, df: pd.DataFrame, data_frame_name: str, scores: dict, news_validation_df: pd.DataFrame):
    for i, alg in enumerate(algs):
        get_result(alg, algs_names[i], vectorizer, vectorizer_name, df, data_frame_name, scores, news_validation_df)

def get_result(alg: str, alg_name:str, vectorizer, vectorizer_name: str, df: pd.DataFrame, data_frame_name: str, scores: dict, news_validation_df: pd.DataFrame):

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], shuffle= True, test_size = 0.3, random_state = 42)
    vectorizer.fit(X_train)
    
    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    X_validate = news_validation_df["text"]
    Y_validate = news_validation_df["label"]
    X_validate = vectorizer.transform(X_validate).toarray()
    
    start_train_time = perf_counter_ns()
    alg.fit(X_train, y_train)
    end_train_time = perf_counter_ns()

    start_predict_time = perf_counter_ns()
    y_predict = alg.predict(X_test)
    end_predict_time = perf_counter_ns()

    y_validate = alg.predict(X_validate)
    print(alg_name, vectorizer_name, data_frame_name)
    print(y_validate)

    scores["algorithm"].append(alg_name)
    scores["dataset"].append(data_frame_name)
    scores["vectorizer"].append(vectorizer_name)

    scores_decimals = 4
    scores["accuracy"].append(round(accuracy_score(y_test, y_predict), scores_decimals))
    scores["precision"].append(round(precision_score(y_test, y_predict, pos_label="fake"), scores_decimals))
    scores["recall"].append(round(recall_score(y_test, y_predict, pos_label="fake"), scores_decimals))
    scores["f1"].append(round(f1_score(y_test, y_predict, pos_label="fake"), scores_decimals))
    scores["validation_accuracy"].append(round(accuracy_score(Y_validate, y_validate), scores_decimals))

    time_decimals = 2
    scores["train_time"].append(round((end_train_time - start_train_time)*1e-9, time_decimals))	
    scores["predict_time"].append(round((end_predict_time - start_predict_time)*1e-9, time_decimals))
 

def main():
    pass

if __name__ == "__main__":
    main()
    
    
   

