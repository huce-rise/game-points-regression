import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # Khởi tạo weights và bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            # Dự đoán
            y_pred = np.dot(x, self.weights) + self.bias

            # Tính gradient
            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Cập nhật tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Tính loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)

        return self

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)


def train_model(x, y, preprocessor, config):
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config['model']['test_size'], random_state=config['model']['random_state']
    )

    # Tạo pipeline với LinearRegression tự viết
    model = LinearRegression(
        learning_rate=config['model']['linear_regression']['learning_rate'],
        n_iterations=config['model']['linear_regression']['n_iterations']
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Huấn luyện
    pipeline.fit(x_train, y_train)

    # Dự đoán
    y_pred = pipeline.predict(x_test)

    # Đánh giá
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        'linear_regression': {
            'mse': mse,
            'r2': r2,
            'model': pipeline
        }
    }

    # Lưu model
    joblib.dump(pipeline, 'models/linear_regression.pkl')

    return results, x_test, y_test