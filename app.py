from flask import Flask, render_template, request, send_from_directory
from werkzeug import secure_filename
import os, sys
import argparse
import subprocess 

app = Flask(__name__)

@app.route('/uploader', methods = ['POST'])
def upload_file_1():
   if request.method == 'POST':
	  img = request.get_data()
	  f = open('image.jpg','w')
	  f.write(img)
	  return 'Done'
      

@app.route('/filter', methods = ['POST'])
def upload_filter():
   if request.method == 'POST':
	  img = request.get_data()
	  subprocess.call('python CLI_artistic_CNN_3.0.py %d' % int(img), shell=True)
	  root_dir = app.root_path
	  return send_from_directory(os.path.join(root_dir), 'result_at_iteration_4.png')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', '--address', type=str, help='Address of the distributing server')
	parser.add_argument('-p', '--port', type=int, help='Port used by the distributing server')
	args = parser.parse_args()
	if len(sys.argv)==0:
		app.run(debug=True)
	else:
		app.run(debug=True, host=args.address, port=args.port)
