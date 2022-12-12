from tkinter import*
from newspaper import Article
import nltk 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')
root=Tk()
root.title("Newspaper")
root.geometry('400x400')
my_listbox=Listbox(root)
my_listbox.pack(pady=15)
my_listbox.insert(END,"authors")
my_listbox.insert(END,"publish_date")
my_listbox.insert(END,"top_image")
my_listbox.insert(END,"movies")
my_listbox.insert(END,"title")
my_listbox.insert(END,"keywords")
my_listbox.insert(END,"summary")
my_listbox.insert(END,"parse")
my_listbox.insert(END,"classification")
label=Label(root,text="ici la reponse")
label.pack(pady=15)
entry1=Entry(root,width= 40)
entry1.pack()
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv", sep='\t')
data["category"].value_counts()
data = data[["title", "category"]]

x = np.array(data["title"])
y = np.array(data["category"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train,y_train)



def button_command():
    url=str(entry1.get())
    
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    article.authors
    article.publish_date
    article.top_image
    article.movies
    article.keywords
    article.summary
    article.title
    data1 = cv.transform([article.summary]).toarray()
    output = model.predict(data1)

 
    if my_listbox.get(my_listbox.curselection())=="authors":
        label.config(text=article.authors)
    elif my_listbox.get(my_listbox.curselection())== "publish_date":
        label.config(text=article.publish_date)
    elif my_listbox.get(my_listbox.curselection())== "top_image":
        label.config(text=article.top_image)
    elif my_listbox.get(my_listbox.curselection())== "movies":
        label.config(text=article.movies)
    elif my_listbox.get(my_listbox.curselection())== "keywords":
        label.config(text=article.keywords)
    elif my_listbox.get(my_listbox.curselection())=="summary":
        label.config(text=article.summary)
    elif my_listbox.get(my_listbox.curselection())=="title":
        label.config(text=article.title)
    elif my_listbox.get(my_listbox.curselection())=="parse":
        label.config(text= article.parse())
    elif my_listbox.get(my_listbox.curselection())=="classification":
        label.config(text=output )      
    return None
Button(root,text='valider',command=button_command).pack()
Button(root, text='Quit', command=root.quit).pack()

root.mainloop()




