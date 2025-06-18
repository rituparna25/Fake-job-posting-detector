# 🕵️‍♀️ Fake Job Posting Detector

This project uses **Natural Language Processing (NLP)** and **Logistic Regression** to detect fake job postings based on their text descriptions. The model is trained using a dataset from Kaggle and aims to protect users from scams while job hunting.

---

## 🎯 Aim

To build a **machine learning model** that analyzes job descriptions and predicts whether a job posting is **real or fake**, using text data and basic NLP techniques.

---

## 📁 File Structure

```
fake_job_detection/
│
├── fake_job_postings.csv      # Dataset (Kaggle: Fake Job Postings)
├── fake_job_detector.py       # Python script with model + user input
├── README.md                  # Project documentation (this file)
```

---

## ⚙️ Requirements

Install the following libraries before running the script:

```bash
pip install pandas scikit-learn
```

> Python version: Recommended 3.7+

---

## 🧠 Model Details

- **Vectorization**: CountVectorizer
- **Model**: Logistic Regression
- **Evaluation**: Classification Report

### 📊 Final Model Performance:

| Metric      | Real Jobs (0) | Fake Jobs (1) |
|-------------|---------------|---------------|
| Precision   | 0.99          | 0.94          |
| Recall      | 1.00          | 0.75          |
| F1-Score    | 0.99          | 0.83          |

- **Accuracy**: 98%
- **Macro Avg F1**: 91%

---

## 🚀 How to Run

### Step 1: Place the dataset

Download [`fake_job_postings.csv`](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) from Kaggle and place it in the same directory as your script.

### Step 2: Run the script

```bash
python fake_job_detector.py
```

---

## 📝 Example Input & Output

### 🔍 Input:
```
Enter job details or enter 0 to stop: Earn money fast working from home. No experience needed. Start today!
```

### 🧾 Output:
```
No!! This is Fake job posting
```

---

### ✅ Input:
```
Enter job details or enter 0 to stop: Looking for a software developer with 3 years of experience in Python and Django. Must be able to work with REST APIs and Git.
```

### 🧾 Output:
```
Yayy!! This is a Real job posting
```

---

## 💡 Key Concepts Used

- Text preprocessing (`lower()`, regex cleaning)
- Count Vectorization (`CountVectorizer`)
- Binary classification (`LogisticRegression`)
- Class imbalance handling (evaluated via precision, recall)
- User input testing

---

## 🛠️ Problems Faced

1. **Missing Text Data**:
   - Many job posts had missing fields (`company_profile`, `benefits`).
   - Solved by filling missing values with empty strings instead of dropping.

2. **Class Imbalance**:
   - Fake jobs (`fraudulent=1`) are rare compared to real ones.
   - Addressed with careful model evaluation using F1-score and weighted averages.

3. **Long Vectorization Time**:
   - Large text fields caused `CountVectorizer` to create massive feature sets.
   - Fixed by limiting `max_features=10000`.

---

## 📌 Notes

- This is a **basic text-based model** — it doesn't use advanced deep learning or embeddings.
- It's **not 100% accurate**, but provides good real-time feedback for users.
- Use results as guidance, **not absolute truth** — always verify job offers independently.

---

## 📚 Dataset Source

[Kaggle: Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
