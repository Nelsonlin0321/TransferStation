from flask import Flask, render_template, request,url_for
import requests

app = Flask(__name__)
app.config['TEMPLATE_AUTO_RELOAD'] = True


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    context = {"predict_result":''}
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        text = request.form.get('sentence')
        context = {"input_sentence": text}

        import requests
        r = requests.post(
            "https://api.deepai.org/api/sentiment-analysis",
            data={
                'text': text,
            },
            headers={'api-key': '02c3a9fe-7b99-4a17-a242-1410dd81ec93'}
        )
        print(r.json())
        context = {"input_sentence": text,"predict_result":r.json()['output'][0]}
        return render_template("index.html", **context)


if __name__ == '__main__':
    app.run(debug=True)
