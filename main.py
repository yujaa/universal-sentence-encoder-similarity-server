from flask import Flask, session
from flask import request
from flask import render_template

from datetime import datetime
import numpy as np
import tensorflow_hub as hub
#import tensorflow._api.v2.compat.v1 as tf
import json
import tensorflow as tf
#tf.disable_v2_behavior()

app = Flask(__name__, static_folder="static")

# universal-sentence-encoder module path.
module_url = "./large5"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def embed(input):
  return model(input)

@app.route("/similar", methods=['GET', 'POST'])
def similar():
    print(request.data)
    data = json.loads(request.data)
    print(data["a"])
    messages = data["a"]

    message_embeddings_ = run_and_plot(messages)
    print(message_embeddings_)
    return json.dumps({"value": str(message_embeddings_)})

def get_most_similar_words(labels, features):

    similarity_matrix = np.inner(features, features)
    most_similar_indices = np.argsort(similarity_matrix[0])[::-1]
    most_similar_words = {labels[i]: similarity_matrix[0][i] for i in most_similar_indices[1:]}
    return most_similar_words

def run_and_plot(messages_):
  message_embeddings_ = embed(messages_)
  return get_most_similar_words(messages_, message_embeddings_)

if __name__ == "__main__":
    app.run(host='0.0.0.0')