import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep needed columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels into numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and target
x = data['message']
y = data['label']

# Convert text into numbers
cv = CountVectorizer()
x = cv.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
msg = ["Congratulations! You won a free iPhone"]
msg_count = cv.transform(msg)

prediction = model.predict(msg_count)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")