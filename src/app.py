from flask import Flask
from flask_cors import CORS
from main import main

app = Flask(__name__)
CORS(app)


@app.route("/graph", methods=["GET"])
def sparx():
    """ Serves the json to the localhost/graph route for retrieval by react server """
    return main()

app.run()
