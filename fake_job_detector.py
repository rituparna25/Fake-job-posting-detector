import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def text_cleaning(txt):
    txt = txt.lower()
    txt = re.sub(r'[^a-zA-Z\s]','',txt)
    return txt

# loading the dataset from csv file
df=pd.read_csv("fake_job_postings.csv")

# displaying the first few rows and the columns in the dataset
print(df.head())
print(df.columns)

# here we are cleaning the data
text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
for col in text_columns:
    df[col] = df[col].fillna('')

df['text'] = df['title'] + " " +df["company_profile"] + " " + df["description"] + " " + df["requirements"] + " " + df["benefits"]
df= df[["text","fraudulent"]]

df['text'] = df['text'].apply(text_cleaning)

#here we are vectorizing the 'text' column
vec = CountVectorizer(stop_words='english',max_features=10000)
X = vec.fit_transform(df['text']).toarray()
y=df['fraudulent']

#training and testing the logistic regression model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)

#classification report of the model trained
print("Report for Logistice Regression: ")
print(classification_report(y_test, model.predict(X_test)))

while(True):
    print("Welcome to Fake Job Posting detection system\n")
    print("Caution : This model may make mistakes.")
    text = input("Enter job details or enter 0 to stop: ")
    if(text == "0"):
        break
    new_text = text.lower()
    new_text = re.sub(r'[^a-zA-Z\s]','',text)
    vector = vec.transform([new_text])
    pred = model.predict(vector)
    print("\nNo!! This is Fake job posting\n" if pred[0]==1 else "\nYayy!! This is a Real job posting\n")
    


