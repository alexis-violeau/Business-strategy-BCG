from src import loading, metrics, modeling, preprocessing, resampling, evaluation

df = loading.load_data()
df_preprocess = preprocessing.preprocess_data(df)
X_train, X_val, y_train, y_val = preprocessing.split(df_preprocess)
model = modeling.fit_model(X_train,y_train)
evaluation.evaluate_model(model,X_val,y_val)