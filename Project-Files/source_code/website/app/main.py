import subprocess
import os
from flask import Flask, render_template, jsonify, request, send_from_directory

#global variable- keeps track of the line # from char_lstm_poems.txt
#lineNum = 0

# instance of the Flask class is our WSGI application
app = Flask(__name__)

@app.route('/')
def index():
    #render html template
    return render_template('index.html')

@app.route('/poem')
def get_poem():
    # Run sample.py to get poem output from model
	#subprocess.call("python sample.py models/Bidirectional_1509158787.h5", shell=True)
    
    # Return text from char_lstm_poems.txt
    #return send_from_directory('./results/', 'Bidirectional_1509158787.txt')
	#with open('/app/results/Bidirectional_1509158787.txt', 'r') as f:
	#with handles opening and closing a file (file automatically closes after exiting from with execution block)
	#https://stackoverflow.com/questions/2424000/read-and-overwrite-a-file-in-python
	with open('results/char_lstm_poems.txt', 'r') as f: #/app/results/char_lstm_poems.txt
	#results = f.read()
	#f.seek(0)
		lines = f.readlines()
	with open('results/line_number.txt', 'r+') as f:
		lineNum = f.read()
		lineNum = int(lineNum)
		endPoem = lineNum + 16
		results = ''.join(lines[lineNum:endPoem])
		f.seek(0)
		if endPoem >= 797: #797 lines in char_lstm_poems.txt, reset to 0 if it gets to the end
			endPoem = 0
		endPoem = str(endPoem) #need to convert to string before writing to file
		f.write(endPoem)
		f.truncate() #overwrite previous
	print results
			
	return render_template('index.html', results=results)

@app.route('/usefulLinks')
def useful_links():
	#render html template
	return render_template('useful_links.html')

@app.route('/about')
def about():
	#render html template
	return render_template('about.html')


if __name__ == "__main__":
    # for debugging purposes
    app.run(host='0.0.0.0', debug=True, port=8000)
