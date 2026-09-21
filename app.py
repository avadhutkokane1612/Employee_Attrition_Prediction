from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Features
model = pickle.load(open("models/employee_attrition_model.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))


# Category Encoding Mapping
mapping = {

    "BusinessTravel": {
        "Non-Travel": 0,
        "Travel_Frequently": 1,
        "Travel_Rarely": 2
    },

    "Department": {
        "Human Resources": 0,
        "Research & Development": 1,
        "Sales": 2
    },

    "EducationField": {
        "Human Resources": 0,
        "Life Sciences": 1,
        "Marketing": 2,
        "Medical": 3,
        "Other": 4,
        "Technical Degree": 5
    },

    "Gender": {
        "Female": 0,
        "Male": 1
    },

    "OverTime": {
        "No": 0,
        "Yes": 1
    },

    "MaritalStatus": {
        "Divorced": 0,
        "Married": 1,
        "Single": 2
    },

    "JobRole": {
        "Healthcare Representative": 0,
        "Human Resources": 1,
        "Laboratory Technician": 2,
        "Manager": 3,
        "Manufacturing Director": 4,
        "Research Director": 5,
        "Research Scientist": 6,
        "Sales Executive": 7,
        "Sales Representative": 8
    }
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    input_data = []

    for feature in features:

        if feature in mapping:
            value = mapping[feature][request.form[feature]]
        else:
            value = float(request.form[feature])

        input_data.append(value)


    # Convert into numpy array
    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)


    if prediction[0] == 1:
        result = "Employee is likely to leave the company"
    else:
        result = "Employee is likely to stay in the company"


    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)