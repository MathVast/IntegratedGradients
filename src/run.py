from integrated_gradients import predict
from flask import Flask, jsonify, request, flash, render_template, render_template_string
from markupsafe import Markup

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def get_prediction():
    query = request.form['query']
    passage = request.form['passage']

    error = None

    if not query or not passage:
        error = "Please provide both query and passage."

    if error is not None:
        flash(error)

    html_text = predict(
        query,
        passage,
        20,
        10
    )
    return render_template('index.html', attribution=Markup(html_text))


if __name__ == '__main__':
    app.run()
