from flask import Flask, request, jsonify  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import pandas as pd

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

@api.route('/score')
class score(Resource):
    def post(self):  # POST 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        data = pd.read_csv('sentiment.csv')
        return data.to_json(orient='records')

if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=True)
