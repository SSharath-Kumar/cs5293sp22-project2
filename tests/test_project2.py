import predictor
import json


def test_prediction():
    knn_input = 3
    ingredients = ['plain flour', 'eggs', 'milk', 'salt', 'wheat']

    output = predictor.predict_cuisine(knn_input, ingredients)

    valid_output = False
    try:
        json.loads(output)
        valid_output = True
    except:
        valid_output = False

    assert valid_output
