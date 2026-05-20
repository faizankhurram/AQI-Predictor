"""
Optional TensorFlow MLP trainer.
Only used if it meaningfully beats the sklearn best model (lower avg RMSE).
"""

import logging
import os

import numpy as np
import joblib

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")


def build_mlp(input_dim: int, output_dim: int):
    """Shallow MLP: two hidden layers with dropout."""
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_dim),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_mlp(X_train, Y_train, X_test, Y_test, epochs: int = 50, batch_size: int = 64):
    """
    Train MLP and return (model, test_predictions).
    X/Y expected as numpy arrays.
    """
    from sklearn.preprocessing import StandardScaler
    from src.models.metrics import evaluate_all_horizons
    from src.features.build_features import get_target_columns

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    Y_train_s = scaler_y.fit_transform(Y_train)

    model = build_mlp(X_train.shape[1], Y_train.shape[1])

    import tensorflow as tf
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]
    model.fit(
        X_train_s, Y_train_s,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=0,
    )

    Y_pred_s = model.predict(X_test_s)
    Y_pred = scaler_y.inverse_transform(Y_pred_s)

    target_cols = get_target_columns()
    true_dict = {col: Y_test[:, i] for i, col in enumerate(target_cols)}
    pred_dict = {col: Y_pred[:, i] for i, col in enumerate(target_cols)}
    metrics = evaluate_all_horizons(true_dict, pred_dict)
    log.info("TF-MLP avg RMSE=%.2f MAE=%.2f R²=%.3f",
             metrics["average"]["rmse"],
             metrics["average"]["mae"],
             metrics["average"]["r2"])

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "tf_mlp.keras")
    model.save(model_path)
    scaler_x_path = os.path.join(MODELS_DIR, "tf_scaler_x.pkl")
    scaler_y_path = os.path.join(MODELS_DIR, "tf_scaler_y.pkl")
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)
    log.info("TF model saved → %s", model_path)

    return model, metrics, scaler_x, scaler_y
