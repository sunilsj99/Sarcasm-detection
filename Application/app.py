from flask import Flask, render_template, request
import spacy
import pickle

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')


def preprocess_text_and_predictions(news, mapping={0: 'No Sarcasm', 1: 'Sarcasm Detected'}):
    clf = pickle.load(open('model.pkl', 'rb'))
    tokens = [token.lemma_ for token in nlp(
        news) if not token.is_punct | token.is_space | token.is_stop]
    ar = [' '.join(tokens)]
    pred = clf.predict(ar)
    return mapping[pred[0]]


@app.route('/')
def index():
    img = "../static/bOs_gzNA_400x400.jpeg"
    return render_template("index.html", image=img)


@app.route('/result', methods=['GET', 'POST'])
def result():
    if(request.method == 'GET'):
        img = "../static/CAB2F17A-6F5E-4BE8-84D00FCA9AC7A2C6_source.jpg"
        res = "Errrr !! You can't come here without submiiting the form..."
        return render_template("result.html", image=img, message=res)
    else:
        news = request.form.get("news")
        res = preprocess_text_and_predictions(news)
        if(res == "No Sarcasm"):
            img = "../static/0549f9323ab332601ae29a08eff19e92dc8da6-wide-thumbnail.jpg"
        else:
            img = "../static/54869c33dc275506f587e2c6df255f45.jpg"
        return render_template("result.html", image=img, message=res)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
