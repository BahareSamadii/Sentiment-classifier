# Sentiment-classifier
### V1: 
##Vectorize the text data
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
##Train the SVM classifier
#classifier = SVC(kernel='linear')
#Accuracy: 0.4904

## V2:
#Changing the number of max_features decrease the accuracy

## V3:
#Model Selection and Tuning
#Define the parameter grid for GridSearchCV
#param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
#grid_search = GridSearchCV(SVC(), param_grid, cv=5)
#grid_search.fit(X_train_vec, y_train)
#best_params = grid_search.best_params_
#Train the SVM classifier with best parameters
#classifier = SVC(**best_params)
#classifier.fit(X_train_vec, y_train)
#Accuracy: 0.4522

