from flask import Flask, render_template, url_for, request
from views import views
import pickle
import pandas as pd, numpy as np

clf = pickle.load(open('svc_b.pkl', 'rb'))

app = Flask(__name__)
app.register_blueprint(views, url_prefix='/')

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		me = request.form['message']
		message = [float(x) for x in me.split()]
		vect = np.array(message).reshape(1, -1)
		try:
			my_prediction = clf.predict(vect)
		except(ValueError):
			return('You must put in 30 integer values in the container. Try again')
	return render_template('result.html',prediction = my_prediction)

if __name__ =='__main__':
    app.run(debug=True, port=8000)
