from flask import Flask, request, jsonify  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import pandas as pd

#nltk.download('punkt')
app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

# url pattern으로 code 설정
@api.route('/sentiment/<string:code>')
class sentiment(Resource):
    def post(self,code):  # POST 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        data = pd.read_csv('news_senti_ratio.csv')
        send_data = {
            "result": {
                "code": code,
                "positive ratio":int(data[data["Stock"]==code]["positive"]),
                "negative ratio":int(data[data["Stock"]==code]["negative"]),
            }
        }
        return jsonify(send_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=True)
