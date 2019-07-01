import tkinter
from tkinter import *
from tkinter import messagebox

import sklearn
from PIL import ImageTk, Image
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

top = tkinter.Tk()
top.geometry("592x744")  # Width x Height
top.title("Cricket World Cup 2019 Winner")

load = Image.open("C:/Users/dania/Pictures/cw20192.jpg")
render = ImageTk.PhotoImage(load)

img = Label(top, image=render)
img.image = render
img.place(x=0, y=0)


def winner(x):
    if x.Result == 1:
        x["Winning_Team"] = x.Team1
    else:
        x["Winning_Team"] = x.Team2
    return x


def winner_two_teams(team1, team2, x):
    x = x[(x["Team1"] == team1) & (x["Team2"] == team2) | (x["Team1"] == team2) & (x["Team2"] == team1)]

    x = x.apply(winner, axis=1)
    return x


def Naive_Bayes():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Required Data
    data = pd.read_csv(
        "C:/Users/dania/Desktop/Course Folder/2nd Half/DataMining/worldcup-2019-prediction/worldcup 2019 prediction/Data_Training.csv")
    data.dropna(inplace=True)
    data_2019 = pd.read_csv(
        "C:/Users/dania/Desktop/Course Folder/2nd Half/DataMining/worldcup-2019-prediction/worldcup 2019 prediction/New_Data_Testing_2019.csv")
    data_2019.drop(columns=["Unnamed: 0"], inplace=True)

    x = data.drop(columns=["Output"])
    x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
    y = data.Output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
    x_2019 = data_2019.drop(columns=["Match", "Team1", "Team2"])

    from sklearn.naive_bayes import GaussianNB

    # Create a Gaussian Classifier
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    from sklearn import metrics
    prediction_2019 = clf.predict(x_2019)
    data_2019["Result"] = prediction_2019
    data_2019_final = data_2019.apply(winner, axis=1)
    results_2019 = data_2019_final.groupby("Winning_Team").size()
    results_2019 = results_2019.sort_values(ascending=False)
    print(results_2019)
    print(data_2019)

    messagebox.showinfo("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


def Naive_Bayes2(a,b):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print(a)
    print(b)
    # Required Data
    data = pd.read_csv(
        "C:/Users/dania/Desktop/Course Folder/2nd Half/DataMining/worldcup-2019-prediction/worldcup 2019 prediction/Data_Training.csv")
    data.dropna(inplace=True)
    data_2019 = pd.read_csv(
        "C:/Users/dania/Desktop/Course Folder/2nd Half/DataMining/worldcup-2019-prediction/worldcup 2019 prediction/New_Data_Testing_2019.csv")
    data_2019.drop(columns=["Unnamed: 0"], inplace=True)

    x = data.drop(columns=["Output"])
    x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
    y = data.Output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
    x_2019 = data_2019.drop(columns=["Match", "Team1", "Team2"])

    from sklearn.naive_bayes import GaussianNB

    # Create a Gaussian Classifier
    clf = GaussianNB()

    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict Output
    y_pred = clf.predict(x_test)

    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?

    prediction_2019 = clf.predict(x_2019)
    data_2019["Result"] = prediction_2019

    print(winner_two_teams(a,b, data_2019))

    messagebox.showinfo("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

def RandomForest():
    import keras
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Required Data
    data = pd.read_csv("Data_Training.csv")
    data.dropna(inplace=True)
    data_2019 = pd.read_csv("New_Data_Testing_2019.csv")
    data_2019.drop(columns=["Unnamed: 0"], inplace=True)

    x = data.drop(columns=["Output"])
    x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
    y = data.Output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
    x_2019 = data_2019.drop(columns=["Match", "Team1", "Team2"])

    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    prediction_2019 = clf.predict(x_2019)
    data_2019["Result"] = prediction_2019

    data_2019_final = data_2019.apply(winner, axis=1)
    results_2019 = data_2019_final.groupby("Winning_Team").size()
    results_2019 = results_2019.sort_values(ascending=False)
    print(results_2019)
    print(data_2019)
    print(winner_two_teams("India", "Pakistan", data_2019))


def RandomForest2(a,b):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Required Data
    data = pd.read_csv("Data_Training.csv")
    data.dropna(inplace=True)
    data_2019 = pd.read_csv("New_Data_Testing_2019.csv")
    data_2019.drop(columns=["Unnamed: 0"], inplace=True)

    x = data.drop(columns=["Output"])
    x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
    y = data.Output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
    x_2019 = data_2019.drop(columns=["Match", "Team1", "Team2"])

    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    prediction_2019 = clf.predict(x_2019)
    data_2019["Result"] = prediction_2019
    print(winner_two_teams(a, b, data_2019))



def NeuralNetwork():
    # -*- coding: utf-8 -*-
    """
    Created on Sun Dec 16 17:13:06 2018
    @author: prohi
    """
    import keras
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Required Data
    data = pd.read_csv("Data_Training.csv")
    data.dropna(inplace=True)
    data_2019 = pd.read_csv("New_Data_Testing_2019.csv")
    data_2019.drop(columns=["Unnamed: 0"], inplace=True)

    x = data.drop(columns=["Output"])
    x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
    y = data.Output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
    x_2019 = data_2019.drop(columns=["Match", "Team1", "Team2"])

    # Converting categorical data to categorical
    num_categories = 2
    y_train = keras.utils.to_categorical(y_train, num_categories)
    y_test = keras.utils.to_categorical(y_test, num_categories)

    # Model Building
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(50, activation="relu", input_dim=41))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(80, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(2, activation="softmax"))

    # Compiling the model - adaDelta - Adaptive learning
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Training and evaluating
    batch_size = 50
    num_epoch = 1000
    model_log = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                          validation_data=(x_test, y_test))

    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Train accuracy:', train_score[1])
    print('Test accuracy:', test_score[1])

    # Predictions for the 2019 World Cup
    prediction_2019 = model.predict_classes(x_2019)
    data_2019["Result"] = prediction_2019


    data_2019_final = data_2019.apply(winner, axis=1)
    results_2019 = data_2019_final.groupby("Winning_Team").size()
    results_2019 = results_2019.sort_values(ascending=False)
    print(results_2019)
    print(data_2019)



def NeuralNetwork2(a,b):
    # -*- coding: utf-8 -*-
    """
    Created on Sun Dec 16 17:13:06 2018
    @author: prohi
    """
    import keras
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Required Data
    data = pd.read_csv("Data_Training.csv")
    data.dropna(inplace=True)
    data_2019 = pd.read_csv("New_Data_Testing_2019.csv")
    data_2019.drop(columns=["Unnamed: 0"], inplace=True)

    x = data.drop(columns=["Output"])
    x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
    y = data.Output
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)
    x_2019 = data_2019.drop(columns=["Match", "Team1", "Team2"])

    # Converting categorical data to categorical
    num_categories = 2
    y_train = keras.utils.to_categorical(y_train, num_categories)
    y_test = keras.utils.to_categorical(y_test, num_categories)

    # Model Building
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(50, activation="relu", input_dim=41))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(80, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(2, activation="softmax"))

    # Compiling the model - adaDelta - Adaptive learning
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Training and evaluating
    batch_size = 50
    num_epoch = 1000
    model_log = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                          validation_data=(x_test, y_test))

    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Train accuracy:', train_score[1])
    print('Test accuracy:', test_score[1])

    # Predictions for the 2019 World Cup
    prediction_2019 = model.predict_classes(x_2019)
    data_2019["Result"] = prediction_2019

    print(winner_two_teams(a, b, data_2019))



def Home():
    import tkinter

    top = tkinter.Tk()
    top.geometry("450x300+500+300")  # Width x Height
    top.title("Predict Winner of World Cup")

    B = tkinter.Button(top, width=20, bg="blue", text="Naive_Bayes1", command=Naive_Bayes, relief='flat', padx=15, pady=5)
    B.place(x=50, y=60)
    B = tkinter.Button(top, width=20, bg="blue", text=" Random Forest", command=RandomForest, relief='flat', padx=15,
                       pady=5)
    B.place(x=250, y=60)
    B2 = tkinter.Button(top, width=20, bg="blue", text="Neural Network", command=Naive_Bayes, relief='flat',
                       padx=15, pady=5)
    B2.place(x=130, y=100)

def Home2():
    import tkinter

    top = tkinter.Tk()
    top.geometry("600x300")  # Width x Height
    top.title("Predict Winner of World Cup")

    Lb1 = tkinter.Listbox(top,exportselection=0,selectmode=SINGLE)
    Lb1.insert(1, "Afghanistan")
    Lb1.insert(2, "Australia")
    Lb1.insert(3, "Bangladesh")
    Lb1.insert(4, "England")
    Lb1.insert(5, "India")
    Lb1.insert(6, "New Zealand")
    Lb1.insert(7, "Pakistan")
    Lb1.insert(8, "South Africa")
    Lb1.insert(9, "Sri Lanka")
    Lb1.insert(10, "West Indies")

    Lb1.pack()
    Lb1.place(x=50, y=20)

    a = Lb1.get(ANCHOR)


    Lb2 = tkinter.Listbox(top,exportselection=0,selectmode=MULTIPLE)
    Lb2.insert(1, "Afghanistan")
    Lb2.insert(2, "Australia")
    Lb2.insert(3, "Bangladesh")
    Lb2.insert(4, "England")
    Lb2.insert(5, "India")
    Lb2.insert(6, "New Zealand")
    Lb2.insert(7, "Pakistan")
    Lb2.insert(8, "South Africa")
    Lb2.insert(9, "Sri Lanka")
    Lb2.insert(10, "West Indies")

    Lb2.pack()
    Lb2.place(x=400, y=20)


    b = Lb2.get(ANCHOR)


    B = tkinter.Button(top, width=20, bg="blue", text="Naive_Bayes", command=lambda:Naive_Bayes2(a,b), relief='flat', padx=15, pady=5)
    B.place(x=10, y=200)

    B2 = tkinter.Button(top, width=20, bg="blue", text=" Random Forest", command=lambda:RandomForest2("England","Bangladesh"), relief='flat', padx=15,
                       pady=5)
    B2.place(x=210, y=200)
    B3 = tkinter.Button(top, width=20, bg="blue", text="Neural_Network", command=lambda:NeuralNetwork2(a,b), relief='flat',
                       padx=15, pady=5)
    B3.place(x=410,y=200)

def LoadDataset():
   import tkinter.ttk as ttk
   import csv
   root = Tk()
   root.title("Datasets")
   width = 770
   height = 700
   screen_width = root.winfo_screenwidth()
   screen_height = root.winfo_screenheight()
   x = (screen_width / 2) - (width / 2)
   y = (screen_height / 2) - (height / 2)
   root.geometry("%dx%d+%d+%d" % (width, height, x, y))
   root.resizable(0, 0)

   TableMargin = Frame(root, width=1000)
   TableMargin.pack(side=TOP)
   scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
   scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
   tree = ttk.Treeview(TableMargin,
                       columns=( "Mat_avg", "Inns_Bat_avg", "NO_avg", "Runs_Bat_avg", "HS_avg", "HS_max", "Ave_Bat_avg","Ave_Bat_max","Inns_Bowl_avg","Balls_avg","Match", "Team1","Team2"),
                       height=400, selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
   scrollbary.config(command=tree.yview)
   scrollbary.pack(side=RIGHT, fill=Y)
   scrollbarx.config(command=tree.xview)
   scrollbarx.pack(side=BOTTOM, fill=X)

   tree.heading('Mat_avg', text="Mat_avg", anchor=W)
   tree.heading('Inns_Bat_avg', text="Inns_Bat_avg", anchor=W)
   tree.heading('NO_avg', text="NO_avg", anchor=W)
   tree.heading('Runs_Bat_avg', text="insu", anchor=W)
   tree.heading('HS_avg', text="HS_avg", anchor=W)
   tree.heading('HS_max', text="HS_max", anchor=W)
   tree.heading('Ave_Bat_avg', text="Ave_Bat_avg", anchor=W)
   tree.heading('Ave_Bat_max', text="Ave_Bat_max", anchor=W)
   tree.heading('Inns_Bowl_avg', text="Inns_Bowl_avg", anchor=W)
   tree.heading('Balls_avg', text="Balls_avg", anchor=W)
   tree.heading('Match', text="Match", anchor=W)
   tree.heading('Team1', text="Team1", anchor=W)
   tree.heading('Team2', text="Team2", anchor=W)

   tree.column('#0', stretch=NO, minwidth=0, width=0)
   tree.column('#1', stretch=NO, minwidth=0, width=80)
   tree.column('#2', stretch=NO, minwidth=0, width=80)
   tree.column('#3', stretch=NO, minwidth=0, width=80)
   tree.column('#4', stretch=NO, minwidth=0, width=80)
   tree.column('#5', stretch=NO, minwidth=0, width=80)
   tree.column('#6', stretch=NO, minwidth=0, width=80)
   tree.column('#7', stretch=NO, minwidth=0, width=80)
   tree.column('#8', stretch=NO, minwidth=0, width=80)
   tree.column('#9', stretch=NO, minwidth=0, width=80)
   tree.column('#10', stretch=NO, minwidth=0, width=80)
   tree.column('#11', stretch=NO, minwidth=0, width=80)
   tree.column('#12', stretch=NO, minwidth=0, width=80)
   tree.column('#13', stretch=NO, minwidth=0, width=80)

   tree.pack()
   with open('Data_Testing_2019.csv') as f:
      reader = csv.DictReader(f, delimiter=',')
      for row in reader:
         age = row['Mat_avg']
         plas = row['Inns_Bat_avg']
         pres = row['NO_avg']
         skin = row['Runs_Bat_avg']
         insu = row['Runs_Bat_avg.1']
         mass = row['HS_avg']
         pedi = row['HS_max']
         thalach = row['Ave_Bat_avg']
         exang = row['Ave_Bat_max']
         oldpeak = row['Inns_Bowl_avg']
         slope = row['Balls_avg']
         ca = row['Match']
         thal = row['Team1']
         classAttribute = row['Team2']

         tree.insert("", 0, values=( plas, pres, skin, insu, mass, pedi, thalach,exang,oldpeak,slope ,ca ,thal ,classAttribute ))
   if __name__ == '__main__':
      root.mainloop()


def ROC():

   from sklearn.metrics import auc, roc_curve
   import pandas as pd

   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import GaussianNB
   from sklearn.ensemble import RandomForestClassifier

   data = pd.read_csv('Data_Training.csv')
   data.dropna(inplace=True)
   x = data.drop(columns=["Output"])
   x = data.drop(columns=["Output", "Unnamed: 0", "match_url"])
   y = data.Output
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=64)

   MLA = [
      sklearn.naive_bayes.GaussianNB(),
      sklearn.ensemble.RandomForestClassifier(n_estimators=100),]
   index = 1
   for alg in MLA:
      predicted = alg.fit(x_train, y_train).predict(x_test)
      fp, tp, th = roc_curve(y_test, predicted)
      roc_auc_mla = auc(fp, tp)
      MLA_name = alg.__class__.__name__
      plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)' % (MLA_name, roc_auc_mla))

      index += 1

   plt.title('ROC Curve comparison')
   plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
   plt.plot([0, 1], [0, 1], 'r--')
   plt.xlim([0, 1])
   plt.ylim([0, 1])
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.show()


B = tkinter.Button(top, width=20, bg="white", text="Predict Winner of World Cup", command=Home, relief='flat', padx=15, pady=5)
B.place(x=200, y=430)

B1 = tkinter.Button(top, width=20, bg="white", text="Predicit Winner of one Match", command=Home2, relief='flat', padx=15, pady=5)
B1.place(x=200, y=500)

B2 = tkinter.Button(top, width=20, bg="white", text="View DataSet", command=LoadDataset, relief='flat', padx=15, pady=5)
B2.place(x=200, y=570)

B3 = tkinter.Button(top, width=20, bg="white", text="Visualization", command=ROC, relief='flat', padx=15, pady=5)
B3.place(x=200, y=640)

top.mainloop()