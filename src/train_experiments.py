import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    MODEL_DIR,
    RESULTS_DIR,
    SUMMARY_CSV,
    CLASS_INDICES_PATH,
    CNN_MODEL_PATH,
    EFF_MODEL_PATH,
    RESNET_MODEL_PATH,
    EPOCHS_CNN,
    EPOCHS_STAGE_1,
    EPOCHS_STAGE_2
)

from data_utils import make_generators
from model_zoo import build_cnn, build_efficientnet, build_resnet
from evaluate_utils import evaluate_predictions


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_class_weight(train_gen):

    y_train = train_gen.classes

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    return dict(enumerate(class_weights_arr))


def compile_model(model, lr=1e-3, label_smoothing=0.05):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),

        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),

        metrics=["accuracy"]
    )

    return model


def build_callbacks(model_path):

    return [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]


def merge_history(h1, h2):

    merged = {}

    for k in h1.history.keys():
        merged[k] = h1.history[k] + h2.history[k]

    return type(
        "History",
        (),
        {"history": merged}
    )()


def train_one_experiment(exp):

    model_type = exp["model_type"]
    strong_aug = exp["strong_aug"]
    use_class_weight = exp["use_class_weight"]
    fine_tune = exp["fine_tune"]
    exp_name = exp["name"]

    print(f"\n========== {exp_name} ==========")

    # =====================================================
    # LOAD DATA
    # =====================================================

    train_gen, val_gen, test_gen = make_generators(
        model_type=model_type,
        strong_aug=strong_aug
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    # save class indices
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:

        json.dump(
            train_gen.class_indices,
            f,
            ensure_ascii=False,
            indent=4
        )

    # =====================================================
    # CLASS WEIGHT
    # =====================================================

    class_weight = None

    if use_class_weight:
        class_weight = get_class_weight(train_gen)

    # =====================================================
    # CNN
    # =====================================================

    if model_type == "cnn":

        model = build_cnn(num_classes)

        model = compile_model(
            model,
            lr=5e-4,
            label_smoothing=0.1
        )

        if strong_aug:
            model_path = CNN_MODEL_PATH
        else:
            model_path = CNN_MODEL_PATH.replace(
                ".keras",
                f"_{exp_name}.keras"
            )

        callbacks = build_callbacks(model_path)

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_CNN,
            class_weight=class_weight,
            callbacks=callbacks
        )

    # =====================================================
    # EFFICIENTNET
    # =====================================================

    elif model_type == "effnet":

        model, base_model = build_efficientnet(
            num_classes=num_classes,
            trainable_backbone=False
        )

        # ---------------- STAGE 1 ----------------

        model = compile_model(
            model,
            lr=1e-3,
            label_smoothing=0.1
        )

        if fine_tune:
            model_path = EFF_MODEL_PATH
        else:
            model_path = EFF_MODEL_PATH.replace(
                ".keras",
                f"_{exp_name}.keras"
            )

        callbacks = build_callbacks(model_path)

        history_stage1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_STAGE_1,
            class_weight=class_weight,
            callbacks=callbacks
        )

        # ---------------- STAGE 2 ----------------

        if fine_tune:

            print("\nFine-tuning EfficientNet...\n")

            base_model.trainable = True

            # FINE-TUNING MẠNH HƠN
            for layer in base_model.layers[:-80]:
                layer.trainable = False

            model = compile_model(
                model,
                lr=5e-6,
                label_smoothing=0.1
            )

            history_stage2 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS_STAGE_2 + 5,
                class_weight=class_weight,
                callbacks=callbacks
            )

            history = merge_history(
                history_stage1,
                history_stage2
            )

        else:
            history = history_stage1

    # =====================================================
    # RESNET50
    # =====================================================

    elif model_type == "resnet":

        model, base_model = build_resnet(
            num_classes=num_classes,
            trainable_backbone=False
        )

        # ---------------- STAGE 1 ----------------

        model = compile_model(
            model,
            lr=1e-3,
            label_smoothing=0.1
        )

        if fine_tune:
            model_path = RESNET_MODEL_PATH
        else:
            model_path = RESNET_MODEL_PATH.replace(
                ".keras",
                f"_{exp_name}.keras"
            )

        callbacks = build_callbacks(model_path)

        history_stage1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_STAGE_1,
            class_weight=class_weight,
            callbacks=callbacks
        )

        # ---------------- STAGE 2 ----------------

        if fine_tune:

            print("\nFine-tuning ResNet50...\n")

            base_model.trainable = True

            # FINE-TUNING MẠNH HƠN
            for layer in base_model.layers[:-70]:
                layer.trainable = False

            model = compile_model(
                model,
                lr=5e-6,
                label_smoothing=0.1
            )

            history_stage2 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS_STAGE_2 + 5,
                class_weight=class_weight,
                callbacks=callbacks
            )

            history = merge_history(
                history_stage1,
                history_stage2
            )

        else:
            history = history_stage1

    else:
        raise ValueError(
            "model_type phải là cnn, effnet hoặc resnet"
        )

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print("\nLoading best saved model...")
    model = tf.keras.models.load_model(model_path)

    # compile lại sau khi load model
    model = compile_model(model)

    # =====================================================
    # EVALUATE
    # =====================================================

    print("\nEvaluating on test set...")

    test_gen.reset()

    y_prob = model.predict(
        test_gen,
        verbose=1
    )

    y_true = test_gen.classes

    out_prefix = os.path.join(
        RESULTS_DIR,
        exp_name
    )

    result = evaluate_predictions(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        out_prefix=out_prefix
    )

    # =====================================================
    # RESULT ROW
    # =====================================================

    row = {

        "experiment": exp_name,

        "model_type": model_type,

        "strong_aug": strong_aug,

        "use_class_weight": use_class_weight,

        "fine_tune": fine_tune,

        "accuracy": result["accuracy"],

        "precision_weighted":
            result["precision_weighted"],

        "recall_weighted":
            result["recall_weighted"],

        "f1_weighted":
            result["f1_weighted"],

        "top3_accuracy":
            result["top3_accuracy"],

        "best_val_accuracy":
            float(np.max(history.history["val_accuracy"]))
    }

    print("\nRESULT:")
    print(row)

    return row


def main():

    ensure_dir(MODEL_DIR)
    ensure_dir(RESULTS_DIR)

    experiments = [

        # =================================================
        # CNN
        # =================================================

        {
            "name": "cnn_weak_nocw",
            "model_type": "cnn",
            "strong_aug": False,
            "use_class_weight": False,
            "fine_tune": False
        },

        {
            "name": "cnn_strong_cw",
            "model_type": "cnn",
            "strong_aug": True,
            "use_class_weight": True,
            "fine_tune": False
        },

        # =================================================
        # EFFICIENTNET
        # =================================================

        {
            "name": "eff_weak_nocw",
            "model_type": "effnet",
            "strong_aug": False,
            "use_class_weight": False,
            "fine_tune": False
        },

        {
            "name": "eff_strong_cw_ft",
            "model_type": "effnet",
            "strong_aug": True,
            "use_class_weight": True,
            "fine_tune": True
        },

        # =================================================
        # RESNET50
        # =================================================

        {
            "name": "res_weak_nocw",
            "model_type": "resnet",
            "strong_aug": False,
            "use_class_weight": False,
            "fine_tune": False
        },

        {
            "name": "res_strong_cw_ft",
            "model_type": "resnet",
            "strong_aug": True,
            "use_class_weight": True,
            "fine_tune": True
        }
    ]

    rows = []

    start_time = time.time()

    for exp in experiments:

        row = train_one_experiment(exp)

        rows.append(row)

        df = pd.DataFrame(rows)

        df.to_csv(
            SUMMARY_CSV,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nCurrent Summary:")
        print(df)

    total_time = (
        time.time() - start_time
    ) / 60

    print("\n===================================")
    print("TRAINING FINISHED")
    print("===================================")

    print(f"Saved summary to: {SUMMARY_CSV}")
    print(f"Total time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()