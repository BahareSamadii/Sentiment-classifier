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

## V4:
# Build the neural network model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_vec.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train_vec, y_train, validation_data=(X_test_vec, y_test), epochs=20, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = model.predict(X_test_vec)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = np.mean(np.array_equal(y_pred_classes, np.argmax(y_test, axis=1)))
print(f"Accuracy: {accuracy:.4f}")
#Accuracy: 0.49
