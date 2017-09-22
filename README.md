<div align="center">
  <img src="https://github.com/denver1117/classificator/blob/master/doc/logo/main_logo.png"><br>
</div>

-----------------

# Classificator : a convenience wrapper for classification methods in Python's scikit-learn

### About
The Classificator is a wrapper around scikit-learn’s classification suite, whose primary function is to normalize and automate the process by which common machine learning classification models are trained and validated. This is specifically in the context of text feature spaces, which require NLP techniques for preprocessing and vectorization. There is an additional focus on group cross validation, where labeled groups exist as data features, and variance is known to be smaller within groups. 

### Features
Load a dataset 
- User points to a dataset locally or in Amazon S3 
- User defines simple data specs 
- User defines feature columns, label column and group columns 

Load a configuration 
- User specifies preprocessing options, model choices, and hyperparameter grids per model choice 
- User specifies meta parameters 

Train and validate various model choices 
- Uses common cross validation and grid search techniques, and chooses the best model based on a user supplied scoring option 
- Return validation statistics and save the model 
- User defines output location for validation report and pickled model object 

