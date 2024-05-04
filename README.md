# Social Media Rumour Detection

## Overview
This project aims to detect rumors in social media posts using machine learning techniques. It involves preprocessing the text data, vectorizing it, training various classification models, and evaluating their performance.

## Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- nltk
- gensim

## Language Used
Python

## Preprocessing
### Description
The text data in the 'Description' column of the dataset is preprocessed before feeding it into the models. The preprocessing steps include:
- Converting text to lowercase
- Tokenizing text
- Removing punctuation and special characters
- Removing stopwords
- Lemmatizing tokens

### Implementation
The preprocessing is implemented using Python libraries such as NLTK and regular expressions. The preprocess_text function is applied to the 'Description' column of the DataFrame.

## Vectorization
### Description
Text data is converted into numerical form using vectorization techniques. Three main vectorization methods are used:
- TF-IDF Vectorization: Converts text data into TF-IDF (Term Frequency-Inverse Document Frequency) vectors.
- Bag of Words (CountVectorizer): Converts text data into frequency-based vectors.
- Word2Vec: Converts text data into dense word embeddings using the Word2Vec algorithm.

### Implementation
- TF-IDF vectorization is performed using the TfidfVectorizer from scikit-learn.
- Bag of Words vectorization is performed using the CountVectorizer from scikit-learn.
- Word2Vec embedding is trained using the Word2Vec model from the Gensim library.

## Models
### Description
Several classification models are trained to classify social media posts as rumors or non-rumors. The following models are used:
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- Gradient Boosting

### Implementation
The models are trained using the vectorized data. The performance of each model is evaluated using accuracy and F1-score metrics.

## Model Training and Evaluation
### Description
The dataset is split into training and testing sets. Each model is trained on the training set and evaluated on the testing set.

### Implementation
- The train_test_split function from scikit-learn is used to split the dataset.
- Each model is trained and tested using the respective vectorized data.
- Accuracy and F1-score metrics are computed for each model.

## Hyperparameter Tuning
### Description
Hyperparameter tuning is performed to optimize the performance of the SVM model.

### Implementation
- GridSearchCV is used to search for the best combination of hyperparameters for the SVM model.
- The best-performing SVM model is selected based on cross-validated accuracy.

## Visualization
### Description
The accuracy and F1-score of each model are visualized for comparison.

### Implementation
- Matplotlib is used to create bar charts showing the performance of each model. Separate visualizations are created for TF-IDF and Bag of Words vectorization methods.

## Result and Comparison
### Description
The performance of each model is compared based on accuracy and F1-score metrics.

### Result
- SVM Accuracy: 0.6498
- Naive Bayes Accuracy: 0.6179
- Random Forest Accuracy: 0.6341
- Gradient Boosting Accuracy: 0.5906

## State of the Tweet Analysis
### Description
The 'State of Tweet' column indicates the nature of each social media post, as determined by an agreement between annotators.

### Analysis
The distribution of different states of tweets in the dataset is as follows:
- Rumor Posts (r): 1020
- Anti-Rumor Posts (a): 3024
- Question Posts (q): 49
- Non-related Posts (n): 4517

## How to Run the Code
To run the code on Google Colab, follow these steps:
1. Upload the dataset (DATASET_R1.xlsx) to your Google Drive.
2. Open the notebook file (Social_Media_Rumour_Detection.ipynb) in Google Colab.
3. Mount your Google Drive by running the code cell containing the appropriate command.
4. Modify the file path in the code to point to the dataset in your Google Drive.
5. Execute each code cell sequentially to preprocess the data, train the models, and evaluate their performance.

## Conclusion
In conclusion, this project demonstrates the effectiveness of machine learning techniques in detecting rumors in social media posts. By preprocessing text data, vectorizing it, and training classification models, we can accurately classify social media posts as rumors or non-rumors.
