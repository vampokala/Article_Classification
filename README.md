# Article_Classification

# Case Study: Articles Categorization

This project focuses on categorizing articles into predefined categories using machine learning models. The notebook explores various text processing techniques and employs different models to achieve the best performance in terms of accuracy and recall.

## Project Overview

In this case study, we aim to develop a model that can automatically categorize articles into specific categories such as sports, politics, news, business, entertainment, and health. This can be particularly useful for media companies looking to organize and streamline content distribution.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Modeling](#modeling)
   - [Text Vectorization](#text-vectorization)
   - [Random Forest Classifier](#random-forest-classifier)
   - [Model Tuning](#model-tuning)
3. [Model Evaluation](#model-evaluation)
4. [Conclusion](#conclusion)

## Data Preprocessing

### Steps:
- **Text Cleaning:** The raw text data is cleaned to remove noise and irrelevant information. This includes removing punctuation, converting text to lowercase, and stripping whitespace.
- **Tokenization:** The cleaned text is then tokenized, splitting the text into individual words or tokens.
- **Stop Words Removal:** Common stop words that do not contribute significantly to the meaning are removed.
- **Stemming/Lemmatization:** Words are reduced to their base or root form.

## Modeling

### Text Vectorization

We employed two popular word embedding techniques to convert text data into numerical format suitable for machine learning models:

1. **Word2Vec:** Generates word vectors by training a shallow neural network to predict a word based on its context.
2. **GloVe (Global Vectors for Word Representation):** Produces word embeddings by aggregating global word-word co-occurrence statistics from a corpus.

### Random Forest Classifier

The primary model used in this case study is the **Random Forest Classifier**. This model is an ensemble of decision trees that improves prediction accuracy and controls over-fitting.

### Model Tuning

To enhance the performance of the Random Forest model, we used **GridSearchCV** to fine-tune hyperparameters. The tuning focused on parameters such as:
- Number of trees in the forest
- Maximum depth of each tree
- Minimum samples required to split a node
- Minimum samples required at each leaf node

## Model Evaluation

We evaluated the performance of our models using the following metrics:
- **Accuracy:** The proportion of correctly classified instances out of the total instances.
- **Recall:** The ability of the model to identify all relevant instances.
- **F1 Score:** The harmonic mean of precision and recall.

The best performing model was the **Random Forest with Word2Vec embeddings**, achieving:
- **Accuracy:** 86%
- **Recall:** 86%

This model was able to categorize articles with high accuracy, particularly excelling in categories with ample training data such as sports, politics, news, and business.

## Conclusion

- We successfully built and tuned a Random Forest model to categorize articles based on their text content.
- The model demonstrated robust performance, especially in categories with a significant amount of training data.
- Categories with fewer samples, like entertainment and health, showed lower performance, indicating a need for more data.
- The final model can be deployed to automatically categorize future articles, aiding in efficient content management.

## Future Work

- Collect more samples for underrepresented categories to improve model performance.
- Experiment with other advanced models like **BERT** or **GPT** to potentially enhance accuracy.
- Implementing real-time deployment for automatic categorization in a production environment.

## How to Use

To reproduce the results or apply the model to new data:
1. Clone the repository:
   ```bash
   git clone https://github.com/vampokala/Article_Classification.git
   ```
2. Navigate to the project directory and install dependencies:
   ```bash
   cd articles-categorization
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Case_Study_Articles_Categorization.ipynb
   ```


