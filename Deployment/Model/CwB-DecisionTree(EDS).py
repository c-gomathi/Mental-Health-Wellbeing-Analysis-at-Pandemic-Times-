#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import  variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[2]:


#cwb is the short form for child well being
df_cwb = pd.read_excel("C:/Users/Hp/Desktop/360-DS/Live Project/Data.xlsx")

#Replacing the columns with spaces by underscores for ease of use
df_cwb.columns = df_cwb.columns.str.replace(' ','_')

df_cwb.describe()
df_cwb.info()


# In[3]:


#Demarcation of  Emotional Difficulties subscales and Behaviourial difficulties subscales 
#with logical Column names

#Emotional Difficulties Subscale:EDS
#Behaviourial difficulties Subscale: BDS

df_cwb.columns = df_cwb.columns.str.replace('24._Remember,_there_are_no_right_or_wrong_answers,_just_pick_which_is_right_for_you._','EDS')

df_cwb=df_cwb.rename(columns = {'EDS[I_get_very_angry]':'BDS[I_get_very_angry]',
                                'EDS[I_lose_my_temper]':'BDS[I_lose_my_temper]',
                                'EDS[I_hit_out_when_I_am_angry]':'BDS[I_hit_out_when_I_am_angry]',
                                'EDS[I_do_things_to_hurt_people]':'BDS[I_do_things_to_hurt_people]',
                                'EDS[I_am_calm]':'BDS[I_am_calm]',
                                'EDS[I_break_things_on_purpose]':'BDS[I_break_things_on_purpose]'})

#Mapping of scores as per data dictionary 
#Creation of 2 score lists

scores = {"Never":0,"Sometimes":1,"Always":2}
reverse_score = {"Never":2,"Sometimes":1,"Always":0}

#Score allocation for EDS params
df_cwb['EDS[I_feel_lonely]'].replace(scores,inplace = True)
df_cwb['EDS[I_cry_a_lot]'].replace(scores,inplace = True)
df_cwb['EDS[I_am_unhappy]'].replace(scores,inplace = True)
df_cwb['EDS[I_feel_nobody_likes_me]'].replace(scores,inplace = True)
df_cwb['EDS[I_worry_a_lot]'].replace(scores,inplace = True)
df_cwb['EDS[I_have_problems_sleeping]'].replace(scores,inplace = True)
df_cwb['EDS[I_wake_up_in_the_night]'].replace(scores,inplace = True)
df_cwb['EDS[I_am_shy]'].replace(scores,inplace = True)
df_cwb['EDS[I_feel_scared]'].replace(scores,inplace = True)
df_cwb['EDS[I_worry_when_I_am_at_school]'].replace(scores,inplace = True)

#Score allocation for BDS params
df_cwb['BDS[I_get_very_angry]'].replace(scores,inplace = True)
df_cwb['BDS[I_lose_my_temper]'].replace(scores,inplace = True)
df_cwb['BDS[I_hit_out_when_I_am_angry]'].replace(scores,inplace = True)
df_cwb['BDS[I_do_things_to_hurt_people]'].replace(scores,inplace = True)
df_cwb['BDS[I_am_calm]'].replace(reverse_score,inplace = True)
df_cwb['BDS[I_break_things_on_purpose]'].replace(scores,inplace = True)


# Creation of 2 new columns to measure the scores

df_cwb['Emotional_Difficulties_Subcale'] = ''
df_cwb['Behavioural_Difficulties_Subscale'] = ''

#Calculation of scores for Emotional and Behavioural Difficulty Subscales 

df_cwb['Emotional_Difficulties_Subcale'] = df_cwb['EDS[I_feel_lonely]'] + df_cwb['EDS[I_cry_a_lot]'] + df_cwb['EDS[I_am_unhappy]'] + df_cwb['EDS[I_feel_nobody_likes_me]'] + df_cwb['EDS[I_worry_a_lot]']
+ df_cwb['EDS[I_have_problems_sleeping]'] + df_cwb['EDS[I_wake_up_in_the_night]'] + df_cwb['EDS[I_am_shy]'] + df_cwb['EDS[I_feel_scared]'] + df_cwb['EDS[I_worry_when_I_am_at_school]']

df_cwb['Behavioural_Difficulties_Subscale'] = df_cwb['BDS[I_get_very_angry]']+df_cwb['BDS[I_lose_my_temper]']+df_cwb['BDS[I_hit_out_when_I_am_angry]'] + df_cwb['BDS[I_do_things_to_hurt_people]']+df_cwb['BDS[I_am_calm]']+df_cwb['BDS[I_break_things_on_purpose]']

#Creation of Me and My Feelings column.
#Me and My Feelings is summation of all the 16 columns starting from 24.1
#As per data dictionary: Me and My Feeling measure is an estimation of Child's mental health

df_cwb['Me_and_My_Feelings'] = df_cwb['Emotional_Difficulties_Subcale'] + df_cwb['Behavioural_Difficulties_Subscale']

#Score categorization or discretization of Emotional and Behavioural Difficulties subscales

score = [-1,9,11,15]
values = ['Expected','Borderline_Difficulties','Clinically_Singificant_Difficulties']

df_cwb['Emotional_Difficulties_Subcale'] = pd.cut(df_cwb['Emotional_Difficulties_Subcale'],bins=score,labels = values)
df_cwb['Emotional_Difficulties_Subcale'].value_counts()


score1 = [-1,5,6,20]    
df_cwb['Behavioural_Difficulties_Subscale'] = pd.cut(df_cwb['Behavioural_Difficulties_Subscale'],bins=score1,labels = values)
df_cwb['Behavioural_Difficulties_Subscale'].value_counts()


# In[4]:


#Determine the total null values and remove.
df_cwb.isnull().sum()
df_cwb = df_cwb.dropna(axis = 0, inplace = False)

#Dropping of un-necessary columns
df_cwb.drop(columns = ['EDS[I_feel_lonely]','EDS[I_cry_a_lot]','EDS[I_am_unhappy]',
                                'EDS[I_feel_nobody_likes_me]','EDS[I_worry_a_lot]','EDS[I_have_problems_sleeping]',
                                'EDS[I_wake_up_in_the_night]','EDS[I_am_shy]','EDS[I_feel_scared]','EDS[I_worry_when_I_am_at_school]',
                                'BDS[I_get_very_angry]','BDS[I_lose_my_temper]','BDS[I_hit_out_when_I_am_angry]','BDS[I_do_things_to_hurt_people]','BDS[I_am_calm]','BDS[I_break_things_on_purpose]','ID'], inplace = True)
                     
#Type of family
df_cwb.columns = df_cwb.columns.str.replace('How_many_people_live_in_your_home_with_you_(including_adults)?','Family_Type')

df_cwb['Family_Type'] = ''
Family_Type = {1:"Small",2:"Small",3:"Moderate",4:"Moderate",5:"Large",'6+':"Large"}

df_cwb["Family_Type"] = df_cwb['Family_Type(including_adults)?'].map(Family_Type)

df_cwb.drop(columns = ['Family_Type(including_adults)?'],axis = 1, inplace = True)
        
#Age
#for ease of age calculation we have only considered the year of birth and subtracted from current year
#Month and year are ignored to avoid the complexity

df_cwb['Current_Year'] = 2022
df_cwb['Age'] = df_cwb['Current_Year']-df_cwb['Year']

df_cwb.drop(columns = ['Year','Month','Day','Current_Year'],axis = 1, inplace = True)

df_cwb['Age'].value_counts()

#breakfast
df_cwb.columns = df_cwb.columns.str.replace('1._What_did_you_eat_for_breakfast_YESTERDAY?','Breakfast_Details')

df_cwb['Breakfast_Details?'].value_counts()

df_cwb['Breakfast_Details?'].unique()
df_cwb['Breakfast_Details?'].drop_duplicates()


calorie_type = {"Healthy cereal e.g. porridge, weetabix, readybrek, muesli, branflakes, cornflakes":"Moderate_Calorie",
                "Sugary cereal e.g. cocopops, frosties, sugar puffs, chocolate cereals":"High_Calorie",
                "Cooked breakfast":"Moderate_Calorie","Healthy cereal e.g. porridge, weetabix, readybrek, muesli, branflakes, cornflakes;Fruit":"Moderate_Calorie",
                "waffles with chocolate spread":"High_Calorie","Sausage sandwich":"Low_Calorie",
                "Sausage roll":"Moderate_Calorie","Crumpets with marmite":"High_Calorie",
                "waffles":"High_Calorie","Toast":"Low_Calorie","Tostie sandwich":"Moderate_Calorie","Fruit;Cooked breakfast":"Moderate_Calorie",
                "Bread roll":"Low_Calorie","chocolate spread on bread":"High_Calorie","Nothing;Fruit":"Low_Calorie","Nothing":"Low_Calorie"}


df_cwb['Breakfast_Type'] = df_cwb['Breakfast_Details?'].map(calorie_type)
df_cwb['Breakfast_Type'] = df_cwb['Breakfast_Type'].fillna('Moderate_Calorie')
        
        
df_cwb.drop(columns = ['Breakfast_Details?'],axis = 1, inplace = True)


Days = {"0 days":0,"1-2 days":2,"3-4 days":4,"5-6 days":6,"7 days":7}


#Physical Activites
df_cwb['Physical_Activity_LW'] = df_cwb['6._In_the_last_7_days,_how_many_days_did_you_do_sports_or_exercise_for_at_least_1_hour_in_total._This_includes_doing_any_activities_(including_online_activities)_or_playing_sports_where_your_heart_beat_faster,_you_breathed_faster_and_you_felt_warmer?'].map(Days)

df_cwb.drop(columns = ['6._In_the_last_7_days,_how_many_days_did_you_do_sports_or_exercise_for_at_least_1_hour_in_total._This_includes_doing_any_activities_(including_online_activities)_or_playing_sports_where_your_heart_beat_faster,_you_breathed_faster_and_you_felt_warmer?'],axis = 1, inplace = True)

#TV or Online Games

df_cwb['TV_or_Online_Games'] = df_cwb['7._In_the_last_7_days,_how_many_days_did_you_watch_TV/play_online_games/use_the_internet_etc._for_2_or_more_hours_a_day_(in_total)?'].map(Days)

df_cwb.drop(columns = ['7._In_the_last_7_days,_how_many_days_did_you_watch_TV/play_online_games/use_the_internet_etc._for_2_or_more_hours_a_day_(in_total)?'],axis = 1, inplace = True)

#Tired
df_cwb['Tired_LW'] = df_cwb['8._In_the_last_7_days,_how_many_days_did_you_feel_tired?'].map(Days)

df_cwb.drop(columns = ['8._In_the_last_7_days,_how_many_days_did_you_feel_tired?'], axis = 1, inplace = True)

#Attention
df_cwb['Attention_to_School_Work'] = df_cwb['9._In_the_last_7_days,_how_many_days_did_you_feel_like_you_could_concentrate/pay_attention_well_on_your_school_work?'].map(Days)
df_cwb.drop(columns = ['9._In_the_last_7_days,_how_many_days_did_you_feel_like_you_could_concentrate/pay_attention_well_on_your_school_work?'],axis = 1, inplace = True)


#Aerated Drinks
df_cwb['Aerated_Drink_LW'] = df_cwb['10._In_the_last_7_days,_how_many_days_did_you_drink_at_least_one_fizzy_drink_(e.g._coke,_sprite,_thumsup)?'].map(Days)

df_cwb.drop(columns = ['10._In_the_last_7_days,_how_many_days_did_you_drink_at_least_one_fizzy_drink_(e.g._coke,_sprite,_thumsup)?'],axis = 1, inplace = True)

#Sugary Snacks
df_cwb['Sugary_Snack_LW'] = df_cwb['11._In_the_last_7_days,_how_many_days_did_you_eat_at_least_one_sugary_snack_(e.g._chocolate_bar,_sweets)?'].map(Days)
df_cwb.drop(columns = ['11._In_the_last_7_days,_how_many_days_did_you_eat_at_least_one_sugary_snack_(e.g._chocolate_bar,_sweets)?'],axis = 1, inplace = True)


#Chinese Takeway
df_cwb['Chinese_Takeaway_LW'] = df_cwb['12._In_the_last_7_days,_how_many_days_did_you_eat_take_away_foods_(e.g._Chinese_takeaway)?'].map(Days)
df_cwb.drop(columns = ['12._In_the_last_7_days,_how_many_days_did_you_eat_take_away_foods_(e.g._Chinese_takeaway)?'], axis = 1, inplace = True)


#Type of Play Places 
Play_Area = {"In my house":"Indoor","In my house;Basket ball in house":"Indoor","In my house;In my backyard not garden":"Indoor",
             "In my garden":"Outdoor","In my garden;":"Outdoor","In a place with bushes, trees and flowers;In the woods near my house;Somewhere with water or sand in it":"Outdoor"}

df_cwb['Type_Of_Play_Area'] = df_cwb['19._What_type_of_places_do_you_play_in?'].map(Play_Area)

df_cwb['Type_Of_Play_Area'] = df_cwb['Type_Of_Play_Area'].fillna('Both')

df_cwb['Type_Of_Play_Area'].value_counts()

df_cwb.drop(columns = ['19._What_type_of_places_do_you_play_in?'],axis = 1, inplace = True)

#Mode of contact
df_cwb.columns = df_cwb.columns.str.replace('27._If_yes,_how_are_you_keeping_in_touch_(tick_all_you_use)?','Mode_of_contact')


# Sleep Hours
df_cwb.columns = df_cwb.columns.str.replace('4._What_time_did_you_fall_asleep_YESTERDAY_(to_the_nearest_half_hour)?','Sleep_Time_Yesterday')
df_cwb.columns = df_cwb.columns.str.replace('5._What_time_did_you_wake_up_TODAY_(to_the_nearest_half_hour)?','Wake_Time_Today')

am_pm_conv = {"10:00pm":10,"9:30pm":9.5,"9:00pm":9,"10:30pm":10.5,
              "8:30pm":8.5,"11:00pm":11,"8:00pm":8,"12:00am":12,"12:30am":12.5,
              "2:00am":2,"7:30pm":7.5,"7:00pm":7,"1:00am":1,"1:30am":1.5,
              "6:30pm":6.5,"3:30am":3.5,"3:00am":3,"4:00am":4,"11:30pm":11.5,
              "8:00am":8,"7:30am":7.5,"8:30am":8.5,"9:00am":9,"7:00am":7,
              "9:30am":9.5,"6:30am":6.5,"10:00am":10,"6:00am":6,"10:30am":10.5,
              "11:30am":11.5,"11:00am":11,"5:30am":5.5,"5:00am":5}


df_cwb['Sleep_Time_Yesterday(to_the_nearest_half_hour)?'].value_counts()

df_cwb['Sleep_Time_Yesterday(to_the_nearest_half_hour)?'].replace(am_pm_conv,inplace = True)

df_cwb['Wake_Time_Today(to_the_nearest_half_hour)?'].value_counts()

df_cwb['Wake_Time_Today(to_the_nearest_half_hour)?'].replace(am_pm_conv,inplace = True)

df_cwb['Total_Sleep'] = df_cwb['Wake_Time_Today(to_the_nearest_half_hour)?'] - df_cwb['Sleep_Time_Yesterday(to_the_nearest_half_hour)?']


df_cwb['Total_Sleep']=np.where(df_cwb['Total_Sleep'] < 0,df_cwb.Total_Sleep.add(12),df_cwb.Total_Sleep)

df_cwb['Total_Sleep'].value_counts()

df_cwb.drop(columns = ['Sleep_Time_Yesterday(to_the_nearest_half_hour)?','Wake_Time_Today(to_the_nearest_half_hour)?','What_year_are_you_in_now?','Behavioural_Difficulties_Subscale','Me_and_My_Feelings'],axis = 1, inplace = True)


# In[5]:


#Model Buliding on Emotional Difficulties
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

for column_name in df_cwb.columns:
    if df_cwb[column_name].dtype == object:
        df_cwb[column_name] = lb.fit_transform(df_cwb[column_name])
    else:
        pass

df_cwb['Emotional_Difficulties_Subcale'] = lb.fit_transform(df_cwb['Emotional_Difficulties_Subcale'])


# In[9]:


#Input output split
y = df_cwb.iloc[:,26]
X = df_cwb[cont_cols]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.33)

from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion = 'entropy')
model.fit(X_train,y_train)


# In[10]:


# Prediction on test
pred = model.predict(X_test)


#Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))


# In[11]:


#Prediction on training data
train_pred = model.predict(X_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, train_pred))


# In[13]:


import matplotlib.pyplot as plt, seaborn as sns
features = [i for i in df_cwb.columns if i != "Emotional_Difficulties_Subcale"]
temp = pd.DataFrame({'features': features, 'importance': model.feature_importances_}).sort_values('importance',ascending = False)
temp


# In[14]:


chart = sns.barplot(x = "features",y = "importance",data = temp)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.show()


# In[ ]:


#Interpretations
#From the table and graph above, we can infer that
#1. Children having more friends tend to display Expected or normal Emotional behaviour
#2. Also the frequency of outdoor playing activity  has a fair degree of influence on the emotional aspects.


# In[7]:


def separate(df):
    separated_cols = {
        "categorical" : list(df_cwb.select_dtypes(include = ["bool","object","category"]).columns),
        "continuous" : list(df_cwb.select_dtypes(include = ["int64","float64"]).columns),
        "date" : list(df.select_dtypes(include = ["datetime"]).columns)
    }
    return separated_cols


# In[8]:


cont_cols = separate(df_cwb)["continuous"]
cont_cols


# In[ ]:




