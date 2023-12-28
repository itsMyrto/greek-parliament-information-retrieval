from flask import Flask, render_template, request, redirect, url_for
from search_engine import search_query

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        query = request.form.get('query')

        if query:

            print(f"Received query: {query}")

            results = search_query(query)

            return render_template('result.html', query=query, results=results)

    # Handle the case where the form data is missing or invalid
    print("Invalid request - Missing or invalid form data")
    return "Invalid request"


if __name__ == '__main__':
    app.run(debug=True)