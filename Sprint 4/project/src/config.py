import json

class Config:
    # для воспроизводимости
    SEED = 42

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = ""
    IMAGE_MODEL_UNFREEZE = ""

    # Гиперпараметры
    BATCH_SIZE = 8
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGRESSOR_LR = 1e-3
    EPOCHS = 30
    DROPOUT = 0.3
    HIDDEN_DIM = 1028

    # Пути
    IMAGES_PATH = "data/images"
    TRAIN_DF_PATH = "data/imdb_train.csv"
    VAL_DF_PATH = "data/imdb_val.csv"
    MODELS_PATH = "models"


def serialize_config(config):
    attrs_source = dict(config.__class__.__dict__)
    attrs = {k: v for k, v in attrs_source.items() if not k.startswith("__") and not callable(v)}
    return json.dumps(attrs, indent=4)


def load_config(json_data):
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data 
    config_instance = Config()
    for key, value in data.items():
        setattr(config_instance, key, value)
    return config_instance
