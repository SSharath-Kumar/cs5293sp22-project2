import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import json


def predict_cuisine(knn_inp, user_inp):
    # Reading json file
    df = pd.read_json('yummly.json')

    # Creating Lemmatizer
    lemmatizer = WordNetLemmatizer()

    ingredients = []

    # Lemmatize ingredients used
    for i in range(len(df['ingredients'])):
        row = df['ingredients'].loc[i]
        row_data = ''
        for ing in row:
            row_data += lemmatizer.lemmatize(ing) + ' '

        ingredients.append(row_data)

    # Copy lemmatized ingredients into data frame
    df['Lem ingredients'] = ingredients

    # Creating a TF-IDF vectorizer(1-gram)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), analyzer='word')

    # Vectorized Ingredients from Source Data
    # Used later on for KNN
    knn_x = vectorizer.fit_transform(df['Lem ingredients'])

    # Creating Label Encoder
    le = LabelEncoder()

    # Setting labels and values
    Y = le.fit_transform(df['cuisine'])
    X = vectorizer.fit_transform(df['Lem ingredients'])

    # Split data for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # Creating and training the model
    clf = CalibratedClassifierCV(LinearSVC())
    clf.fit(X_train, Y_train)

    # print('MODEL SCORE: ', clf.score(X_test, Y_test))

    # Lemmatizing User Input
    le_inp = []
    for text in user_inp:
        le_inp.append(lemmatizer.lemmatize(text))

    # Vectorizing User Input
    v_inp = vectorizer.transform(le_inp)

    # Cuisine Prediction
    prediction = clf.predict(v_inp)

    # Inverse Transform to get the cuisine
    pred_cuisine = le.inverse_transform(prediction)[0]

    # Scoring the prediction
    prob_list = np.round_(clf.predict_proba(v_inp)[0], decimals=2)
    pred_score = float(np.amax(prob_list))

    # Finding Similar Ingredients

    # Creating KNN Classifier with user input
    knn = KNeighborsClassifier(n_neighbors=knn_inp)

    cuisine_list = list(df['cuisine'])
    knn.fit(knn_x, cuisine_list)

    # Finding N nearest ingredients using KNN
    distances, ids = knn.kneighbors(v_inp, knn_inp)

    # Storing Distances and IDs
    dist_list = distances[0]
    id_list = ids[0]

    # Lists to hold all the ingredients and IDs from source file
    similar_ingredients_list = []
    ingredient_id_list = []

    # Loop through all the N-nearest meals found using KNN
    for i in range(len(dist_list)):
        idx = id_list[i]

        # Getting Ingredients at the location(s)
        user_ingredient_list = df['ingredients'].loc[idx]

        # Empty string to hold all ingredients for a dish
        row = ''

        # Loop through the list of ingredients
        for ing in user_ingredient_list:
            row += ing + ' '

        ingredient_id_list.append(int(df['id'].loc[idx]))
        similar_ingredients_list.append(row)

    # Empty string - Holds all ingredients passed by user
    ing_str = ''
    for ing in user_inp:
        ing_str += ing + ' '

    # Converting to a list
    user_ingredient_list = [ing_str]
    # Setting inputs for cosine similarity
    sx = vectorizer.fit_transform(similar_ingredients_list)
    sy = vectorizer.transform(user_ingredient_list)
    # Store similarity values
    cs_list = cosine_similarity(sx, sy)
    # Extract similarity values into an array
    similarity_values = []
    for val in cs_list:
        similarity_values.append(val[0])

    # Setting up output dictionary
    d2 = dict(zip(ingredient_id_list, similarity_values))
    d1 = {pred_cuisine: pred_score, 'closest': d2}

    # Writing output to JSON format
    output_json = json.dumps(d1, indent=4)

    return output_json
