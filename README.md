
# Spam Detection in Cybersecurity using Machine Learning

This project is a machine learning-based approach to detecting spam emails. By leveraging natural language processing (NLP) techniques and classification algorithms, the model identifies and categorizes emails as spam or ham (non-spam). The dataset used is from the **SpamAssassin Public Corpus**.

## **Project Overview**

Spam detection plays a crucial role in cybersecurity by filtering unwanted emails and protecting users from phishing and malicious content. This project demonstrates how to preprocess email data, extract features, and train a machine learning model to classify emails accurately.

---

## **Workflow and Code Explanation**

### **Step 1: Import Required Libraries**
The following libraries are used for data handling, feature extraction, and machine learning:
```python
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```
- **os**: For file and directory operations.
- **pandas**: For data manipulation.
- **sklearn**: For feature extraction and classification.

---

### **Step 2: Load the Dataset**
The function `loadingDataSet` reads email data from the spam and ham directories.

```python
def loadingDataSet(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='latin1') as file:
            emails.append(file.read())
    return emails
```
- `folder_path`: Directory path containing spam or ham emails.
- **Returns**: A list of email content.

---

### **Step 3: Specify Dataset Paths**
Adjust these paths to point to your local dataset directories:
```python
spam_path = "./spamassassin-public-corpus/spam"
ham_path = "./spamassassin-public-corpus/ham"
```

---

### **Step 4: Preprocess and Vectorize Data**
The `HashingVectorizer` converts email text into numerical features suitable for machine learning models. The `TfidfTransformer` normalizes the feature values to reduce biases.

```python
vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False)
tfidf_transformer = TfidfTransformer()
```

---

### **Step 5: Train-Test Split**
Data is divided into training and testing sets to evaluate model performance.

```python
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
```

---

### **Step 6: Train the Model**
The **Decision Tree Classifier** is used for email classification:
```python
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
```

---

### **Step 7: Evaluate the Model**
Model accuracy and performance are measured using metrics such as **accuracy** and **confusion matrix**.

```python
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("The accuracy of model is :", accuracy)
print("The Confusion Matrix is:\n", conf_matrix)
```

---

## **Final Output**
The project outputs:
1. **Accuracy Score**: Percentage of correctly classified emails.
2. **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.

---

## **Requirements**
Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

---

## **Usage**
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Add the dataset to the appropriate directories (`spam` and `ham` folders).
3. Run the Jupyter Notebook or Python script:
   ```bash
   python Cybersecurity_final_project.ipynb
   ```

---

## **Dataset**
The dataset is sourced from the [SpamAssassin Public Corpus](http://spamassassin.apache.org/publiccorpus/).

## **For a detailed explanation of the project and the steps involved, check out the full article on Medium:**
https://medium.com/@syedm.upwork/building-a-spam-detection-system-with-machine-learning-3a7e1eb10bc3

