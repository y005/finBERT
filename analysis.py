import sys,nltk,json,requests
from pathlib import Path
from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import nltk

#nltk.download('punkt')
#야후 파이낸스를 통해 특정 종목에 대한 최신 뉴스 기사들의 id 수집
def getStockNews(code):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/list"
    querystring = {"region":"US","snippetCount":"30","s":code}
    payload = ""
    headers = {
        'content-type': "text/plain",
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
        'x-rapidapi-key': "yahoo finance api key"
        }
    response = requests.request("POST", url, data=payload, headers=headers, params=querystring)
    json_data = json.loads(response.text)
    result = []
    for ele in json_data["data"]["main"]["stream"]:
        if len(ele["content"]["finance"]["stockTickers"]) == 1 :#해당 종목에 대한 기사인지 확인
            result.append(ele["content"]["id"])
    return result

#수집한 뉴스 기사 내용 반환
def getNewsContent(ID):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/news/v2/get-details"

    querystring = {"uuid":ID,"region":"US"}
    headers = {
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
        'x-rapidapi-key': "yahoo finance api key"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    json_data1 = json.loads(response.text)
    markup= json_data1["data"]['contents'][0]['content']['body']['markup']
    bs = BeautifulSoup(markup,'html.parser')
    result = bs.find('div', class_ ="caas-body")
    return result.get_text()

#심볼에 해당하는 주식 기사들의 감성비율(긍정/부정)의 판다스 프레임 반환
def sentimentAnalysis(code):
    news_senti_list = []
    dic = {0:"positive",1:"negative"}

    #특정 심볼에 해당하는 종목 기사들 수집
    news = getStockNews(code)
    #기사들의 주요 감성 분석 결과 배열에 저장
    for i in range(len(news)):
        one_news_text = getNewsContent(news[i])
        project_dir = Path.cwd()
        cl_path = '/home/ubuntu/news-senti/models/sentiment/finbert'
        model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
        new_sentences_senti = predict(one_news_text,model)
        #1개의 기사에 대한 주요 감성 확인
        senti = [0,0]
        for j in range(len(new_sentences_senti)):
            if new_sentences_senti["prediction"][j] == "positive":
                senti[0] += 1
            elif new_sentences_senti["prediction"][j] == "negative":
                senti[1] += 1
        news_senti_list.append(dic[senti.index(max(senti))])
    #[심볼,긍정기사비율,부정기사비율] 반환
    ans = [code,news_senti_list.count("positive")/len(news_senti_list)*100,news_senti_list.count("negative")/len(news_senti_list)*100]
    return ans

if __name__=="__main__":
    lists = ["TSLA"]
    #code_list = ["MSFT", "ORCL", "AAPL", "IBM", "GOOGL", "FB", "NFLX", "DIS", "AMZN", "TSLA", "SBUX", "NKE", "WMT", "COST", "KO", "PEP", "V", "PYPL", "BAC", "C", "WFC","JNJ","PFE", "UNH", "AMGN", "LLY","HON", "UNP", "MMM", "TT", "LMT","AMT", "EQIX", "PLD", "O"]

    template = {'Stock' : [], 'positive' : [], 'negative': [] }
    result = pd.DataFrame(template)

    for code in lists:
        senti_analysis = sentimentAnalysis(code)
        update = pd.DataFrame({'Stock' : [senti_analysis[0]], 'positive' : [senti_analysis[1]], 'negative': [senti_analysis[2]]})
        result = pd.concat([result,update])

    result.to_csv("news_senti_ratio.csv",index=False)
