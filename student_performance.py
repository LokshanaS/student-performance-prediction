import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = pd.read_csv("student_data.csv")

print("Dataset:")
print(data)

# Step 2: Convert Pass/Fail to numeric
data['Result'] = data['Result'].map({'Pass': 1, 'Fail': 0})

# Step 3: Separate features and target
X = data[['Attendance', 'Internal_Marks', 'Assignment_Marks']]
y = data['Result']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Step 7: Predict new student
attendance = int(input("\nEnter Attendance (%): "))
internal = int(input("Enter Internal Marks: "))
assignment = int(input("Enter Assignment Marks: "))

new_student = pd.DataFrame(
    [[attendance, internal, assignment]],
    columns=['Attendance', 'Internal_Marks', 'Assignment_Marks']
)

prediction = model.predict(new_student)


if prediction[0] == 1:
    print("\nPrediction Result: PASS")
else:
    print("\nPrediction Result: FAIL")
