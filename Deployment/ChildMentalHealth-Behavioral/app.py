# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:46:25 2022

@author: Deepali Garg
"""

from flask import Flask
import pickle
from flask import render_template
from flask import request
model=pickle.load(open('model.pkl','rb'))



# application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

 # , methods = ['POST']
@app.route('/result' , methods = ['POST'])
def result():
    I_get_very_angry = int(request.form['I_get_very_angry'])
 
    I_lose_my_temper = int(request.form['I_lose_my_temper'])
    I_hit_when_I_am_angry = int(request.form['I_hit_when_I_am_angry'])
    I_do_things_to_hurt_people = int(request.form['I_do_things_to_hurt_people'])
    I_am_calm = int(request.form['I_am_calm'])
    I_break_things_on_purpose = int(request.form['I_break_things_on_purpose'])
    output = model.predict([[I_get_very_angry,I_lose_my_temper,I_hit_when_I_am_angry,I_do_things_to_hurt_people,
                             I_am_calm,I_break_things_on_purpose]])
    if output == '[expected]':
        output = 'expected'
        
    elif output == '[borderline]':
        output = 'borderline'
    else :
        output = 'clinically significant difficulties'
    print(output)
    return render_template('index.html', output = output)


if __name__ == "__main__":
    app.run(debug=True)