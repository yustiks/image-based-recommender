import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask import request

df = pd.read_csv('static/db/train.csv')
big_array = np.load('static/db/big_array.npy')
# load the model from disk
# loaded_model = pickle.load(open('static/db/KNN_model.sav', 'rb'))
from sklearn.neighbors import NearestNeighbors

application = Flask(__name__)


@application.route('/')
def index():
    return render_template('index.html')


@application.route("/sector_picked", methods=('GET', 'POST'))
def recommended_crags():
    if request.method == 'POST':
        # from routes request
        image_id = int(request.form['image_id'])
        metric_name = 'cosine'
        X = np.array(big_array)
        nbrs = NearestNeighbors(n_neighbors=6,
                                algorithm='brute',
                                metric=metric_name
                                ).fit(X)
        distances, neighbours_id = nbrs.kneighbors([X[image_id - 1]], n_neighbors=6, return_distance=True)
        # distances, neighbours_id = loaded_model.kneighbors([X[image_id - 1]],
        #                                                   n_neighbors=6,
        #                                                   return_distance=True)
        distances = distances[0]
        neighbours_id = neighbours_id[0]
        crags_name = []
        sectors_name = []
        for each_id in neighbours_id:
            name_crag = df[df.img_id == each_id + 1].name_crag.values[0]
            crags_name.append(name_crag)
            name_sector = df[df.img_id == each_id + 1].name_sector.values[0]
            sectors_name.append(name_sector)
        result_json = {
            'image_id': image_id,
            'distances': distances[1:],
            'neighbours_id': neighbours_id[1:],
            'crags_name': crags_name[1:],
            'sectors_name': sectors_name[1:]
        }

        return render_template('sector_picked.html', data=result_json)


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8081)
