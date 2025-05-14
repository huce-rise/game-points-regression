from src.data_loader import load_config, load_data
from src.preprocessing import preprocess_data
from src.features import engineer_features
from src.modeling import train_model
from src.utils import save_metrics, plot_loss_curve


def main():
    config = load_config()
    df = load_data(config['data']['raw_path'])
    df = engineer_features(df)
    X, y, preprocessor = preprocess_data(df, config)
    results, X_test, y_test = train_model(X, y, preprocessor, config)
    save_metrics(results, 'results/metrics.txt')

    # Vẽ loss curve
    model = results['linear_regression']['model'].named_steps['model']
    plot_loss_curve(model, 'results/plots/loss_curve.png')


if __name__ == "__main__":
    main()