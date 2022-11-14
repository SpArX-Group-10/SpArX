from main import main
from flask import Flask
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/graph", methods=["GET"])
def sparx():
    return main()

app.run()


