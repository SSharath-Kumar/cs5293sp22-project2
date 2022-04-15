# cs5293sp22-project2

# Cuisine Predictor - Text Analytics Project 2

## Libraries Used

* pandas
* numpy
* nltk
* sklearn
* json

## Assumptions and Bugs 
* There might be a possibility of errors occuring when the program is provided with new ingredients. 

* The probability of the cuisine matching the prediction is considered as the score for the cuisine predicition.

* The score associated with the N nearest meals is assumed to be the similarity between the ingredients provided by the user and the ingredients of the similar meals 


## Functionality

### Models used 
* LinearSVC 
* K-Nearest Neighbours


### Inputs:
* --N (Number of closest meals)
* --ingredients (Ingredients for the meal)

The program takes the inputs through command line arguments. These inputs are passed to the predict cuisine function. 

The yummly.json file is taken as the training dataset for the model. The dataset is read using the pandas library and the ingredients are lemmatized using the *WordNetLemmatizer* from the nltk library.

A *TF-IDF vectorizer* is used to vectorize the lemmatized ingredients. To convert the corresponding cuisines, label encoding is performed using *LabelEncoder* from the sklearn library and the dataset is split into 70-30 ratio using *train_test_split*. The LinearSVC model is created and trained using the training data. 

The user inputs are then processed. The inputs are lemmatized, vectorized and then passed on to the model to perform the prediction. The predicted value is then inversely transformed to get the cuisine label. *The probability of the cuisine matching the prediction is considered as the score*. This score is obtained using the predict_proba function.

#### Finding the N nearest meals matching the given ingredients
A K-Nearest neighbours model is used to find similar meals using the given ingredients. The model is trained using the vectorized ingredients and the corresponding cuisines.

Using the N value provided by the user and the vectorized ingredients, the distances and the row positions for the identical meals are acquired using the KNN model. Using these values, the IDs and the ingredients of the meals are retrieved. The ingredients of the similar meals and the ingredients from user input are vectorized to find the similarity using *cosine_similarity*.

The predicted cuisine, probability score and the closest meals with ID's and the similarity scores are packed in the form of a dictionary. This dictionary is then converted into JSON format using the json library which is displayed as the output using *stdout*
 
## Test Cases

*test_prediction*

A sample input is set in the test case and is passed to the predict cuisine function. The output is stored and valided if it is in the form of a JSON. This validation is done by reading the output using the json library.   

## Steps for local deployment

1] Clone the repository using the below command
git clone https://github.com/SSharath-Kumar/cs5293sp22-project2

2] Install the required dependencies using the command:
pipenv install

## Running the project
Sample command 
pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies" 

## Running the test cases
pipenv run python -m pytest
