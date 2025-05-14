import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, output_path):
    pass

def plot_loss_curve(model, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(model.losses)
    plt.title('Loss Curve (MSE) during Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_metrics(results, output_path):
    with open(output_path, 'w') as f:
        for name, metrics in results.items():
            f.write(f"Model: {name}\nMSE: {metrics['mse']}\nR2: {metrics['r2']}\n\n")