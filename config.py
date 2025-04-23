import os


class Config:
    DATA_PATH_IMAGES = "./data/small_p_data/*.jpg"
    DATA_PATH_MASKS = "./data/merge_pipedata_label/*.jpg"

    # Output directory structure
    OUT_DIR = "./results/segformer"
    MODEL_DIR = os.path.join(OUT_DIR, "models")
    STATS_DIR = os.path.join(OUT_DIR, "stats")
    LOGS_DIR = os.path.join(OUT_DIR, "logs")
    PLOTS_DIR = os.path.join(OUT_DIR, "plots")
    SAMPLES_DIR = os.path.join(OUT_DIR, "samples")

    INPUT_SHAPE = (240, 320, 3)
    N_CLASSES = 1
    BATCH_SIZE = 64

    LEARNING_RATE = 1e-3
    EPOCHS = 10

    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "model_epoch_{epoch:03d}.weights.h5")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
    CSV_PATH = os.path.join(STATS_DIR, "training_history.csv")
    PLOT_PATH = os.path.join(PLOTS_DIR, "training_metrics.png")

    RESUME = False
    MODEL='segformer'
    CHECKPOINT_META_PATH="results/unet/checkpoint_meta.json"
    FineTuned=False

for path in [
    Config.OUT_DIR,
    Config.MODEL_DIR,
    Config.STATS_DIR,
    Config.LOGS_DIR,
    Config.PLOTS_DIR,
    Config.SAMPLES_DIR,
]:
    os.makedirs(path, exist_ok=True)
