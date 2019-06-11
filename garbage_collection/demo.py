import sys
from gensim.models import KeyedVectors
from flask import Flask, request, render_template
import _thread

cn_model = KeyedVectors.load_word2vec_format("./sgns.zhihu.bigram", binary=False)

app = Flask(__name__)


def test():
    while True:
        a = cn_model
        b = cn_model
        a.c = b
        b.c = a


@app.route('/')
def index():
    try:
        _thread.start_new_thread(test, ("train_upload_model",))
    except BaseException as e:
        raise e

    return 'hello world'


if __name__ == '__main__':
    app.run(debug=True)
