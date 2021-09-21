import sys,nltk,json,requests
from pathlib import Path
from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import nltk

nltk.download('punkt')
#야후 파이낸스를 통해 특정 종목에 대한 최신 뉴스 기사 1개 수집
def getStockNews(code):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/list"
    querystring = {"region":"US","snippetCount":"1","s":code}
    payload = ""
    headers = {
        'content-type': "text/plain",
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
        'x-rapidapi-key': "bd2bc7360bmsha9dc79919d1c7e9p1641b7jsnb4bd1c60dbe6"
        }

    response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
    json_data = json.loads(response.text)
    return json_data["data"]["main"]["stream"][0]["content"]["id"]

#수집한 뉴스 기사 내용 반환
def getNewsContent(ID):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/get-details"

    querystring = {"uuid":ID,"region":"US"}
    headers = {
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
        'x-rapidapi-key': "bd2bc7360bmsha9dc79919d1c7e9p1641b7jsnb4bd1c60dbe6"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    json_data1 = json.loads(response.text)
    markup= json_data1["data"]['contents'][0]['content']['body']['markup']
    bs = BeautifulSoup(markup,'html.parser')
    result = bs.find('div', class_ ="caas-body")
    return result.get_text()
# 전체 내용 합친 코드
def sentimentAnalysis(code):
    news = getStockNews(code)
    text = getNewsContent(news)
    project_dir = Path.cwd()
    cl_path = project_dir/'models'/'sentiment'/'finbert'
    model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
    result = predict(text,model)
    result["code"] = code
    result = result[["code","sentence","logit","sentiment_score","prediction"]]
    return result

if __name__=="__main__":
    lists = ["PFE"]
    #code_list = ["MSFT", "ORCL", "AAPL", "IBM", "GOOGL", "FB", "NFLX", "DIS", "AMZN", "TSLA", "SBUX", "NKE", "WMT", "COST", "KO", "PEP", "V", "PYPL", "BAC", "C", "WFC","JNJ","PFE", "UNH", "AMGN", "LLY","HON", "UNP", "MMM", "TT", "LMT","AMT", "EQIX", "PLD", "O"]

    senti = pd.read_csv("sentiment result0.csv")
    for code in lists:
        tmp = sentimentAnalysis(code)
        senti = pd.concat([senti,tmp])

    senti.to_csv("sentiment result.csv",index=False)
