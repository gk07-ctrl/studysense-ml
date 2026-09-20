"""Train and evaluate lightweight StudySense models. No dashboard."""

try:
    from model_utils import run_training
except ImportError:
    from src.model_utils import run_training


def _print_regression(metrics):
    print("regression_test_metrics:")
    for name, values in metrics.items():
        print(
            f"  {name}: "
            f"MAE={values['mae']:.4f} "
            f"RMSE={values['rmse']:.4f} "
            f"R2={values['r2']:.4f}"
        )


def _print_classification(metrics):
    print("classification_test_metrics:")
    for name, values in metrics.items():
        print(
            f"  {name}: "
            f"accuracy={values['accuracy']:.4f} "
            f"precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} "
            f"F1={values['f1']:.4f}"
        )


def main():
    payload, _, _, _ = run_training()
    print("StudySense Phase 3: training and evaluation")
    print(f"n_train: {payload['n_train']}")
    print(f"n_test: {payload['n_test']}")
    print(f"support_needed_threshold: {payload['support_needed_threshold']}")
    print(f"class_counts: {payload['class_counts']}")
    _print_regression(payload["regression"])
    _print_classification(payload["classification"])
    print(f"selected_regression_model: {payload['selected_regression_model']}")
    print(f"selected_classification_model: {payload['selected_classification_model']}")
    print(f"metrics_json: {payload['metrics_path']}")
    print(f"saved_regression_model: {payload['regression_model_path']}")
    print(f"saved_classification_model: {payload['classification_model_path']}")


if __name__ == "__main__":
    main()
