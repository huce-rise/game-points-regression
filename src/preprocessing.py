from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(df, config):
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    X = df.drop(columns=['points'])
    y = df['points']

    cat_cols = config['preprocessing']['categorical_columns']
    num_cols = config['preprocessing']['numerical_columns']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])

    return X, y, preprocessor