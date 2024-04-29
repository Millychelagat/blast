import time
import streamlit as st
import pandas as pd
import pickle as pickle
import os
from streamlit_lottie import st_lottie
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import mysql.connector
import altair as alt

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Alababama@45",
    database="LivestockDiseasesApp"
)
cursor = db_connection.cursor()


def create_users_table():
    cursor.execute("CREATE DATABASE IF NOT EXISTS LivestockDiseasesApp")
    cursor.execute("USE LivestockDiseasesApp")

    cursor.execute("""CREATE TABLE IF NOT EXISTS users (
                            user_id INT AUTO_INCREMENT PRIMARY KEY,
                            user_name VARCHAR(255),
                            user_email VARCHAR(255),
                            user_password VARCHAR(255)
                        )""")

    db_connection.commit()


def create_diseases_table():
    cursor.execute("""CREATE TABLE IF NOT EXISTS diseases (
                        disease_id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT,
                        age INT,
                        gender VARCHAR(10),
                        temperature FLOAT,
                        weight FLOAT,
                        vaccination_status VARCHAR(50),
                        symptom1 VARCHAR(255),
                        symptom2 VARCHAR(255),
                        symptom3 VARCHAR(255),
                        anthrax_prob FLOAT,
                        blackleg_prob FLOAT,
                        foot_and_mouth_prob FLOAT,
                        lumpy_virus_prob FLOAT,
                        pneumonia_prob FLOAT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )""")

    # Commit changes and close connection
    db_connection.commit()


def label_encode_column(data, column):
    lb = LabelEncoder()
    data[column] = lb.fit_transform(data[column])
    return data, lb


def upload_image():
    document_files = st.file_uploader("Upload the Image of the Animal Here : ",
                                      type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
                                      accept_multiple_files=True)
    if document_files is not None:
        save_folder = "images"
        os.makedirs(save_folder, exist_ok=True)

        image_paths = []
        for i, file in enumerate(document_files):
            file_path = os.path.join(save_folder, f"image_{i + 1}.{file.type.split('/')[-1]}")
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(file_path)

        for path in image_paths:
            st.image(path, caption='Image', width=300)

        return image_paths

    if st.button("Upload Image"):
        st.success("Image Uploaded Successfully")


def model():
    df = pd.read_csv("data/animal_disease_dataset.csv")
    df.loc[:, 'Animal'] = 'cow'
    column_to_encode = 'Disease'

    df, encoder = label_encode_column(df, column_to_encode)
    df = df.drop("Animal", axis=1)
    new_features = ['blisters on gums', 'blisters on hooves', 'blisters on mouth',
                    'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound',
                    'depression', 'difficulty walking', 'fatigue', 'lameness', 'loss of appetite',
                    'painless lumps', 'shortness of breath', 'sores on gums', 'sores on hooves',
                    'sores on mouth', 'sores on tongue', 'sweats', 'swelling in abdomen',
                    'swelling in extremities', 'swelling in limb', 'swelling in muscle',
                    'swelling in neck']

    for feature in new_features:
        df[feature] = 0

    for index, row in df.iterrows():
        for symptom_column in ['Symptom 1', 'Symptom 2', 'Symptom 3']:
            symptom = row[symptom_column]
            if symptom in new_features:
                df.loc[index, symptom] = 1

    df.drop(['Symptom 1', 'Symptom 2', 'Symptom 3'], axis=1, inplace=True)

    # Make diseae the last column
    cols = list(df.columns)
    cols.remove('Disease')
    cols.append('Disease')
    df = df[cols]

    # train test split
    X = df.drop("Disease", axis=1)
    Y = df['Disease']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_test, X_train, y_test, y_train = train_test_split(X, Y, test_size=50, random_state=12)
    svc = SVC()
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, pred) * 100
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(svc, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    st.success("Model and Scaler saved successfully")


@st.cache_data()
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


@st.cache_data()
def display_animation():
    lottie_anim = load_lottie("animations/cow_animation.json")
    st_lottie(lottie_anim, speed=1, reverse=False, loop=True, height=200, width=200, quality="high")


def image_model(img):
    CLIENT = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
        api_key="93J2JA4QhOvsFFy3ENF1"
    )

    result = CLIENT.infer(img, model_id="cow-diseae-identifier/1")

    confidence = result.get("confidence")
    top = result.get("top")
    st.write(f"Prediction : {top} with a confidence of {confidence * 100}%")


def auth():
    st.header("CATTLE DISEASE PREDICTOR")
    cred_option = st.selectbox("Do you want to Sign up or Login?", ("Sign Up", "Login"))
    placeholder = st.empty()

    if cred_option == "Sign Up":
        with placeholder.form("signup"):
            st.markdown("#### Enter the Credentials you want to use")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up")
            if submit:
                if username and email and password and password_confirm:
                    if "@" in email:
                        if password == password_confirm:
                            insert_query = "INSERT INTO users (user_name, user_email, user_password) VALUES (%s, %s, %s)"
                            user_data = (username, email, password)
                            cursor.execute(insert_query, user_data)
                            db_connection.commit()
                            st.success("User signed up successfully!")
                            time.sleep(2)
                            placeholder.empty()

                            select_query = "SELECT * FROM users WHERE user_email = %s"
                            cursor.execute(select_query, (email,))
                            user = cursor.fetchone()
                            st.session_state.signed_in = True
                            st.session_state.user_id = user[0]
                            st.session_state.username = user[1]
                            st.rerun()
                        else:
                            st.warning("The Password and Confirm Password Fields do Not Match")
                    else:
                        st.warning("Please Enter a valid Email Address")
                else:
                    st.warning("Please fill in all the necessary form fields")

    elif cred_option == "Login":
        with placeholder.form("login"):
            st.markdown("#### Enter your credentials")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                try:
                    select_query = "SELECT * FROM users WHERE user_email = %s"
                    cursor.execute(select_query, (email,))
                    user = cursor.fetchone()
                    if user and user[3] == password:
                        st.success(f"You are Successfully logged in as {user[1]}")
                        time.sleep(2)
                        placeholder.empty()
                        st.session_state.signed_in = True
                        st.session_state.user_id = user[0]
                        st.session_state.username = user[1]
                        st.rerun()
                    else:
                        st.warning("User Authentication Failed")
                except Exception as e:
                    st.warning(f"Error : {e}")


def display_table_data_users(connection, table_name, column_names):
    if connection:
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            data = cursor.fetchall()
            if data:
                column_mapping = {old_name: new_name for old_name, new_name in zip(cursor.column_names, column_names)}
                renamed_data = [
                    {column_mapping.get(old_name, old_name): value for old_name, value in zip(cursor.column_names, row)}
                    for row in data]

                for row in renamed_data:
                    with st.expander(f"{row[column_names[0]]} - {row[column_names[2]]}"):
                        col1, col2, col3, col4 = st.columns(4)
                        for i, (key, value) in enumerate(row.items()):
                            if i % 4 == 0:
                                col1.write(f"**{key}**: {value}")
                            elif i % 4 == 1:
                                col2.write(f"**{key}**: {value}")
                            elif i % 4 == 2:
                                col3.write(f"**{key}**: {value}")
                            elif i % 4 == 3:
                                col4.write(f"**{key}**: {value}")

                        record_choice = st.radio(f"What do you want to do with the record id {row[column_names[0]]}?",
                                                 ["View", "Update", "Delete"], horizontal=True)
                        if record_choice == "View":
                            pass
                        elif record_choice == "Update":
                            col1_1, col1_2 = st.columns(2)
                            with col1_1:
                                new_username = st.text_input("Enter the New Username", value=row[column_names[1]])
                                new_password = st.text_input("Enter the New Password", type="password")
                            with col1_2:
                                new_email = st.text_input("Enter the New Email Address", value=row[column_names[2]])
                                confirm_new_pass = st.text_input("Confirm the New Password", type="password")
                            confirm_update = st.button(f"Confirm Update Record {row[column_names[0]]}")
                            if confirm_update:
                                try:
                                    if confirm_new_pass == new_password:
                                        sql = """UPDATE users 
                                                     SET user_name = %s, user_email = %s, user_password = %s
                                                     WHERE user_id = %s"""
                                        val = (new_username, new_email, new_password, row[column_names[0]])
                                        cursor.execute(sql, val)
                                        db_connection.commit()
                                        st.success("Record Updated Successfully")
                                    else:
                                        st.warning("The two passwords you entered do not match")
                                except Exception as e:
                                    print(e)
                        elif record_choice == "Delete":
                            st.write("Are you sure you want to delete this record?")
                            cancel_delete = st.button(f"No, Cancel Deletion of id {row[column_names[0]]}")
                            if cancel_delete:
                                pass
                            confirm_delete = st.button(f"Yes, Delete Record id {row[column_names[0]]}")
                            if confirm_delete:
                                sql = "DELETE FROM users WHERE user_id = %s"
                                cursor.execute(sql, (row[column_names[0]],))
                                connection.commit()
                                st.success("User deleted successfully.")

            else:
                st.info("No data available in the table.")
        except Exception as e:
            st.error(f"Error fetching data from the table: {e}")


def display_table_data_diseases(connection, table_name, column_names):
    if connection:
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            data = cursor.fetchall()
            if data:
                column_mapping = {old_name: new_name for old_name, new_name in zip(cursor.column_names, column_names)}
                renamed_data = [
                    {column_mapping.get(old_name, old_name): value for old_name, value in zip(cursor.column_names, row)}
                    for row in data]

                for row in renamed_data:
                    probabilities = [row[column_names[10]], row[column_names[11]], row[column_names[12]],
                                     row[column_names[13]], row[column_names[14]]]
                    highest_prob_index = probabilities.index(max(probabilities))

                    # Mapping disease names to indexes
                    disease_mapping = {
                        0: "anthrax",
                        1: "blackleg",
                        2: "foot and mouth",
                        3: "lumpy virus",
                        4: "pneumonia"
                    }

                    highest_prob_disease = disease_mapping[highest_prob_index]
                    with st.expander(
                            f"{max(probabilities)} % {str(highest_prob_disease)[0].upper() + str(highest_prob_disease)[1:]}"):
                        col1, col2, col3, col4 = st.columns(4)
                        for i, (key, value) in enumerate(row.items()):
                            if i % 4 == 0:
                                col1.write(f"**{key}**: {value}")
                            elif i % 4 == 1:
                                col2.write(f"**{key}**: {value}")
                            elif i % 4 == 2:
                                col3.write(f"**{key}**: {value}")
                            elif i % 4 == 3:
                                col4.write(f"**{key}**: {value}")
                        record_choice = st.radio(f"What do you want to do with the record id {row[column_names[0]]}?",
                                                 ["View", "Delete"], horizontal=True)
                        if record_choice == "Delete":
                            st.write("Are you sure you want to delete this record?")
                            cancel_delete = st.button(f"No, Cancel Deletion of id {row[column_names[0]]}")
                            if cancel_delete:
                                pass
                            confirm_delete = st.button(f"Yes, Delete Record id {row[column_names[0]]}")
                            if confirm_delete:
                                sql = "DELETE FROM diseases WHERE disease_id = %s"
                                cursor.execute(sql, (row[column_names[0]],))
                                connection.commit()
                                st.success("Record deleted successfully.")

            else:
                st.info("No data available in the table.")
        except Exception as e:
            st.error(f"Error fetching data from the table: {e}")


def main():
    st.set_page_config(
        page_title="Livestock Disease Predictor",
        page_icon=":cow2:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    if "signed_in" not in st.session_state:
        st.session_state.signed_in = False
        st.session_state.user_id = ""
        st.session_state.username = ""

    create_users_table()
    create_diseases_table()

    if not st.session_state.signed_in:
        auth()

    if "signed_in" in st.session_state:
        try:
            if st.session_state.signed_in is True:
                if st.session_state.username == "Admin":
                    st.subheader("Admin Panel")
                    admin_choice = st.selectbox("What do you want to View", ("Users", "Diagnosis"))
                    if admin_choice == "Users":
                        column_names = ["User_id", "Username", "User_email", "Password"]
                        display_table_data_users(db_connection, "users", column_names)
                    elif admin_choice == "Diagnosis":
                        column_names = ["Disease_id", "User_id", "Age", "Gender", "Temperature", "Weight",
                                        "Vaccination", "Symptom 1", "Symptom 2", "Symptom 3", "Anthrax Prob.",
                                        "Blackleg Prob.", "Foot and Mouth Prob.", "Lumpy Virus Prob", "Pneumonia Prob"]
                        display_table_data_diseases(db_connection, "diseases", column_names)
                else:
                    animation_col, header_col = st.columns([1, 3])

                    with header_col:
                        st.markdown("<div style='padding-top: 50px;'></div>", unsafe_allow_html=True)
                        st.header("CATTLE DISEASE PREDICTOR")

                    with animation_col:
                        display_animation()

                    with open('cattle_diseases_svc.pkl', 'rb') as model_file:
                        loaded_model = pickle.load(model_file)
                    with open('livestock_scaler.pkl', 'rb') as model_file:
                        loaded_scaler = pickle.load(model_file)

                    age = st.slider('What is the Age of the Animal in Years?', 1, 15)

                    gender = st.selectbox("What is the Gender of the Animal?", ["Male", "Female"])
                    temperature = st.text_input("What is the Temperature of the Animal in Degrees Celsius?")

                    if temperature:
                        temp = float(temperature)
                        temp = (temp * 9 / 5) + 32

                        if temp < 95:
                            st.warning("The entered Temperature is too Low")
                        elif temp > 110:
                            st.warning("The entered Temperature is too High")

                    weight = st.slider("What is the Weight of the Animal in KG?", 0, 500)
                    vaccination_history = st.selectbox("Has the Animal been previously Vaccinated?", ["Yes", "No"])

                    symptom_1 = [
                        "depression",
                        "painless lumps",
                        "loss of appetite",
                        "difficulty walking",
                        "lameness",
                        "chills",
                        "crackling sound",
                        "sores on gums",
                        "fatigue",
                        "shortness of breath",
                        "chest discomfort",
                        "swelling in limb",
                        "swelling in abdomen",
                        "blisters on gums",
                        "swelling in extremities",
                        "swelling in muscle",
                        "blisters on hooves",
                        "blisters on tongue",
                        "sores on tongue",
                        "sweats",
                        "sores on hooves",
                        "blisters on mouth",
                        "swelling in neck",
                        "sores on mouth"
                    ]

                    symptom_2 = [
                        "painless lumps",
                        "loss of appetite",
                        "swelling in limb",
                        "blisters on gums",
                        "depression",
                        "blisters on tongue",
                        "blisters on mouth",
                        "swelling in extremities",
                        "sores on mouth",
                        "lameness",
                        "sores on tongue",
                        "difficulty walking",
                        "sweats",
                        "sores on hooves",
                        "shortness of breath",
                        "crackling sound",
                        "chest discomfort",
                        "chills",
                        "swelling in abdomen",
                        "sores on gums",
                        "swelling in muscle",
                        "fatigue",
                        "swelling in neck",
                        "blisters on hooves"
                    ]

                    symptom_3 = [
                        "loss of appetite",
                        "depression",
                        "crackling sound",
                        "difficulty walking",
                        "painless lumps",
                        "shortness of breath",
                        "lameness",
                        "chills",
                        "swelling in extremities",
                        "fatigue",
                        "chest discomfort",
                        "swelling in limb",
                        "sweats",
                        "blisters on mouth",
                        "sores on mouth",
                        "swelling in abdomen",
                        "blisters on tongue",
                        "swelling in muscle",
                        "swelling in neck",
                        "sores on tongue",
                        "blisters on hooves",
                        "blisters on gums",
                        "sores on hooves",
                        "sores on gums"
                    ]

                    all_symptoms = list(set(symptom_1 + symptom_2 + symptom_3))
                    symptoms = st.multiselect("Select 3 Symptoms", all_symptoms)

                    symptom1_bool = False
                    symptom2_bool = False
                    symptom3_bool = False

                    if st.button("Show Prediction"):

                        if len(symptoms) > 3:
                            st.warning("You have entered More than the 3 Required Symptoms")
                        elif len(symptoms) < 1:
                            st.warning("You have Not entered any Symptoms")
                        elif len(symptoms) < 3:
                            st.warning("You need to enter 3 Symptoms")
                        else:
                            if symptoms:
                                for i in range(len(symptoms)):
                                    if symptoms[i] in symptom_1 and symptom1_bool is False:
                                        symptom1 = symptoms[i]
                                        symptom1_bool = True
                                    elif symptoms[i] in symptom_2 and symptom2_bool is False:
                                        symptom2 = symptoms[i]
                                        symptom2_bool = True
                                    elif symptoms[i] in symptom_3 and symptom3_bool is False:
                                        symptom3 = symptoms[i]
                                        symptom3_bool = True

                            if symptom1_bool and symptom2_bool and symptom3_bool:
                                new_data = {'Age': f"{age}", 'Temperature': f"{temp}", 'Symptom 1': f'{symptom1}',
                                            'Symptom 2': f'{symptom2}', 'Symptom 3': f'{symptom3}'}

                            elif symptom1_bool and symptom2_bool:
                                new_data = {'Age': f"{age}", 'Temperature': f"{temp}", 'Symptom 1': f'{symptom1}',
                                            'Symptom 2': f'{symptom2}', 'Symptom 3': f''}
                            elif symptom1_bool:
                                new_data = {'Age': f"{age}", 'Temperature': f"{temp}", 'Symptom 1': f'{symptom1}',
                                            'Symptom 2': '', 'Symptom 3': ''}

                            new_df = pd.DataFrame([new_data])
                            new_features = ['blisters on gums', 'blisters on hooves', 'blisters on mouth',
                                            'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound',
                                            'depression', 'difficulty walking', 'fatigue', 'lameness',
                                            'loss of appetite',
                                            'painless lumps', 'shortness of breath', 'sores on gums', 'sores on hooves',
                                            'sores on mouth', 'sores on tongue', 'sweats', 'swelling in abdomen',
                                            'swelling in extremities', 'swelling in limb', 'swelling in muscle',
                                            'swelling in neck']

                            for feature in new_features:
                                new_df[feature] = 0

                            for index, row in new_df.iterrows():
                                for symptom_column in ['Symptom 1', 'Symptom 2', 'Symptom 3']:
                                    symptom = row[symptom_column]
                                    if symptom in new_features:
                                        new_df.loc[index, symptom] = 1

                            new_df.drop(['Symptom 1', 'Symptom 2', 'Symptom 3'], axis=1, inplace=True)

                            new_df = loaded_scaler.transform(new_df)
                            svc_model = loaded_model
                            prediction = svc_model.predict(new_df)
                            disease_names = {0: "Anthrax", 1: "Blackleg", 2: "Foot and Mouth", 3: "Lumpy Virus",
                                             4: "Pneumonia"}
                            predicted_disease = disease_names[prediction[0]]

                            st.write(f"Prediction: {predicted_disease}")
                            probabilities = svc_model.predict_proba(new_df)[0] * 100
                            diseases = ["Anthrax", "Blackleg", "Foot and Mouth", "Lumpy Virus", "Pneumonia"]

                            anthrax_prob = round(probabilities[0], 2)
                            blackleg_prob = round(probabilities[1], 2)
                            foot_and_mouth_prob = round(probabilities[2], 2)
                            lumpy_virus_prob = round(probabilities[3], 2)
                            pneumonia_prob = round(probabilities[4], 2)

                            insert_query = """INSERT INTO diseases (user_id, age, gender, temperature, weight, 
                                                                                       vaccination_status, symptom1, symptom2, symptom3, anthrax_prob, 
                                                                                       blackleg_prob, foot_and_mouth_prob, lumpy_virus_prob, pneumonia_prob) 
                                                                                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

                            # Execute the SQL query
                            cursor.execute(insert_query, (
                                st.session_state.user_id, age, gender, temperature, weight, vaccination_history,
                                symptom1, symptom2, symptom3, int(anthrax_prob), int(blackleg_prob),
                                int(foot_and_mouth_prob), int(lumpy_virus_prob), int(pneumonia_prob)))
                            db_connection.commit()

                            data = pd.DataFrame({'Disease': diseases, 'Probability': probabilities})
                            chart = alt.Chart(data).mark_bar().encode(
                                x=alt.X('Disease', axis=alt.Axis(labelAngle=45)),  # Rotate x-axis labels
                                y='Probability',
                                tooltip=['Disease', 'Probability']
                            ).properties(
                                width=alt.Step(80)
                            ).configure_axis(
                                labelFontSize=12
                            ).configure_title(
                                fontSize=20
                            )

                            st.write(chart)

                    img = upload_image()
                    if img:
                        with st.spinner("Processing Image..."):
                            image_model(img)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()