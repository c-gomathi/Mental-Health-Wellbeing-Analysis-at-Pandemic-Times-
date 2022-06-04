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
                                'EDS[I_break_things_on_purpose]':'BDS[I_break_things_on_purpose]',
                               '14._From_your_house,_can_you_easily_walk_to_a_park_(for_example_a_field_or_grassy_area)?':'Can_you_walk_to_a_park',
                               '15._From_your_house,_can_you_easily_walk_to_somewhere_you_can_play?':'Can_you_walk_to_a_play_area',
                               '18._Do_you_have_enough_time_for_play?':'Time_to_play'})

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


# In[4]:


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

df_cwb.drop(columns = ['Sleep_Time_Yesterday(to_the_nearest_half_hour)?','Wake_Time_Today(to_the_nearest_half_hour)?','What_year_are_you_in_now?','Emotional_Difficulties_Subcale','Me_and_My_Feelings'],axis = 1, inplace = True)


# In[5]:


df_cwb['Behavioural_Difficulties_Subscale'].value_counts()


# In[6]:


#Model Buliding on Behavioral Difficulties
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

for column_name in df_cwb.columns:
    if df_cwb[column_name].dtype == object:
        df_cwb[column_name] = lb.fit_transform(df_cwb[column_name])
    else:
        pass

df_cwb['Behavioural_Difficulties_Subscale'] = lb.fit_transform(df_cwb['Behavioural_Difficulties_Subscale'])


# In[7]:


df_cwb['Behavioural_Difficulties_Subscale'].value_counts()


# In[8]:


#Input output split
y = df_cwb.iloc[:,26]
X = df_cwb.iloc[:,df_cwb.columns!='Behavioural_Difficulties_Subscale']
features = [i for i in df_cwb.columns if i != "Behavioural_Difficulties_Subscale"]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[9]:


#Scaling the train data and calculation of VIF score
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

#Calculation of the VIF SCORE
def vif_score(x):
    scaler = StandardScaler()
    arr = scaler.fit_transform(x)
    return pd.DataFrame([[x.columns[i],variance_inflation_factor(arr,i)] for i in range(arr.shape[1])],columns = ["Feature","VIF_SCORE"])                     
    
vif_score(X)
#Since VIF Score for any feature is not above 10, the problem of multi-collinearity does  not exist, 
#so we do not need to remove any columns 
   


# In[10]:


model = LogisticRegression(multi_class = "multinomial", solver = "saga")
model.fit(X_train,y_train)






# In[11]:


# Prediction on test
pred = model.predict(X_test)

#Prediction on training data
train_pred = model.predict(X_train)


# In[12]:


#Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

#Achieved Test accuracy of 94.21


# In[13]:



print(accuracy_score(y_train, train_pred))
#Achieved Train accuracy of 91


# In[14]:


importance = model.coef_[0]
importance = np.sort(importance)

dict(zip(X_train.columns,model.coef_[0]))


# In[15]:


def get_feature_importance(model,feature_names):
    feature_importance = {
        pd.DataFrame(
            {
                'variable':feature_names,
                'coefficient':model.coef_[0]
             }
        )
        .round(decimals = 2) \
        .sort_values('coefficient', ascending=False) \
        .style.bar(color = ['red','green'], align = 'zero')
    }
    return feature_importance


# In[16]:


get_feature_importance(model, X_train.columns)


# In[17]:


#Accuracy Precsion, Recall and F-1 score
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))


# In[18]:


#ROC Curve, ROC Area and area under the curve
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


# In[19]:


# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

#Predicting each class against the other
classifier = OneVsRestClassifier(
    svm.SVC(kernel="linear", probability=True, random_state=random_state)
)
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# In[20]:


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


# In[21]:


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# In[22]:


# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Extension of ROC to multiclass")
plt.legend(loc="lower right")
plt.show()


# In[23]:


#Area under the curve
y_prob = classifier.predict_proba(X_test)

macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
weighted_roc_auc_ovo = roc_auc_score(
    y_test, y_prob, multi_class="ovo", average="weighted"
)
macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
weighted_roc_auc_ovr = roc_auc_score(
    y_test, y_prob, multi_class="ovr", average="weighted"
)
print(
    "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
)
print(
    "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
    "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
)


# In[ ]:





# In[25]:


temp = pd.DataFrame({'variable':features,'coefficient':model.coef_[0]}).sort_values('coefficient', ascending=False)
temp


# In[26]:


chart = sns.barplot(x = "variable",y = "coefficient",data = temp)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.show()


# In[ ]:


#Interpretations
#From the table and graph above, we can infer that
#1. Time to play is feature that has most influence on the Behavioural aspects of the child.Children devouting adequate time
#   to play seem to display normal behavioural traits. 
#2. Location of a child's home near to play area or a park also has a direct impact on the behavioural aspects.
#   Children who can easily walk to a play area from their homes show normal behaviour than children who stay far away from play area

#3  Apart from this, eating habits, academic performance and overall health parameters also influence the behavioural aspects.
#4. Overall, inorder to improve the behavioural aspects special attention needs to be given to physical excercise in the form
#   of play or any other mode along with appropriate diet practises, ensuring all the covid relavant safety measures.

#Conclusion - During the lockdown period, it is quite natural for the child to get lonely and miss contact with friends or miss the play time 
#they used to have if they were in school. Hence until the re-opening of the school , parents should ensure the child is 
#engaging in any form of physical activity along with a healthy diet. 


# In[ ]:


#Input output split on selected features for modelling
y1 = df_cwb.iloc[:,26]
X1 = df_cwb[['Can_you_walk_to_a_play_area','Can_you_walk_to_a_park','Time_to_play']]
#features = [i for i in df_cwb.columns if i = ('Can_you_walk_to_a_play_area','Can_you_walk_to_a_park','Time_to_play')]


# In[ ]:


from sklearn.model_selection import train_test_split
X1_train,X1_test, y1_train,y1_test = train_test_split(X1,y1, test_size = 0.2)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


model1 = LogisticRegression(multi_class = "multinomial", solver = "saga")
model1.fit(X1_train,y1_train)


# In[ ]:


# Prediction on test
pred1 = model1.predict(X1_test)

#Prediction on training data
train_pred1 = model1.predict(X1_train)


# In[ ]:



#Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y1_test, pred1))


# In[ ]:


print(accuracy_score(y1_train, train_pred1))


# In[ ]:


import pickle


# In[ ]:


file = open('log_model1_2.pkl',"wb")
pickle.dump(model1,file)


# In[ ]:


X1_train.shape


# In[ ]:


y1_train.shape


# In[ ]:


X1_test.shape


# In[ ]:


y1_test.shape


# In[ ]:


X1_test.head(5)


# In[ ]:




