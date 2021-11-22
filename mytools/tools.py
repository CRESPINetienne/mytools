def get_pipeline():
    return '''from sklearn.linear_model import model
    from sklearn.model_selection import cross_val_score

    # Preprocessor (prepare the dataset)
    num_transformer = make_pipeline(SimpleImputer(), StandardScaler())
cat_transformer = OneHotEncoder()

preproc = make_column_transformer(
    (num_transformer, make_column_selector(dtype_include=['float64'])),
    (cat_transformer, make_column_selector(dtype_include=['object','bool'])),
    remainder='passthrough')

# Add Estimator
pipe = make_pipeline(preproc, model())

# Train pipeline
pipe.fit(X_train,y_train)

# Make predictions
pipe.predict(X_test.iloc[0:2])

# Score model
pipe.score(X_test,y_test)

# Cross validate pipeline
cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2').mean()'''


from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_punct(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def lower_text(text):
    text = text.lower()
    return text

def remove_numbers(text):
    text = ''.join(char for char in text if not char.isdigit())
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join(w for w in word_tokens if not w in stop_words)
    return text

def lemmat(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens]
    return ' '.join(lemmatized)

def clean_text(df_col):
    df_col = df_col.apply(remove_punct)
    df_col = df_col.apply(lower_text)
    df_col = df_col.apply(remove_numbers)
    df_col = df_col.apply(remove_stopwords)
    df_col = df_col.apply(lemmat)
    return df_col
