# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:46:25 2022

@author: Deepali Garg
"""

from flask import Flask
import pickle
from flask import render_template
from flask import request
model=pickle.load(open('pickle_model%.pkl','rb'))



# application
app = Flask(__name__)

@app.route('/')
def emotinal():
    return render_template('emotinal.html')


 # , methods = ['POST']
@app.route('/result' , methods = ['POST'])
def result():
    going2school = int(request.form['going2school'])
 
    bf_yesterday = int(request.form['bf_yesterday'])
    fruitvegitable_yesterday = int(request.form['fruit&vegitable_yesterday'])
    sports_or_exer = int(request.form['sports_or_exer'])
    feel_tired = int(request.form['feel_tired'])
    attentive_in_school_work = int(request.form['attentive_in_school_work'])
    fizzy_drink = int(request.form['fizzy_drink'])
    sugary_snack = int(request.form['sugary_snack'])
    take_away_food = int(request.form['take_away_food'])
    play_outside = int(request.form['play_outside'])
    playarea = int(request.form['playarea'])
    space_to_relex = int(request.form['space_to_relex'])
    doing_well_with_my_school_work = int(request.form['doing_well_with_my_school_work'])
    part_of_my_school_community = int(request.form['part_of_my_school_community'])
    lots_of_choice_that_are_important_for_me = int(request.form['lots_of_choice_that_are_important_for_me'])
    There_are_lots_of_things_I_am_good_at = int(request.form['There_are_lots_of_things_I_am_good_at'])
    school = int(request.form['school'])
    i_feel_lonely = int(request.form['i_feel_lonely'])
    i_cry_a_lot = int(request.form['i_cry_a_lot'])
    i_m_unhappy = int(request.form['i_m_unhappy'])
    i_feel_nobody_likes_me = int(request.form['i_feel_nobody_likes_me'])
    i_worry_a_lot = int(request.form['i_worry_a_lot'])
    i_have_problems_sleeping = int(request.form['i_have_problems_sleeping'])
    i_wake_up_in_night = int(request.form['i_wake_up_in_night'])
    i_am_shy = int(request.form['i_am_shy'])
    i_feel_scared = int(request.form['i_feel_scared'])
    i_worry_at_school = int(request.form['i_worry_at_school'])
    children_in_your_house_Yes = int(request.form['children_in_your_house_Yes'])
    in_touch_with_friends_Yes = int(request.form['in_touch_with_friends_Yes'])
    output = model.predict([[going2school,bf_yesterday,fruitvegitable_yesterday,sports_or_exer,feel_tired,
                            attentive_in_school_work,fizzy_drink,sugary_snack,take_away_food,play_outside,playarea,
                           space_to_relex,doing_well_with_my_school_work,part_of_my_school_community,
                           lots_of_choice_that_are_important_for_me,There_are_lots_of_things_I_am_good_at,
                           school,i_feel_lonely,i_cry_a_lot,i_m_unhappy,i_feel_nobody_likes_me,
                           i_worry_a_lot,i_have_problems_sleeping,i_wake_up_in_night,i_am_shy,
                           i_feel_scared,i_worry_at_school,children_in_your_house_Yes,in_touch_with_friends_Yes]])
    if output == '[Expected]':
        output = print('Expected')
        
    elif output == '[borderline]':
        output = print('borderline')
    elif output =='[clinically significant difficulties]':
        output = print('clinically significant difficulties')
    print(output)
    return render_template('emotinal.html', output = output)


if __name__ == "__main__":
    app.run(debug=True)    

