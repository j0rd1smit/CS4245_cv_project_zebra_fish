import os

def get_all_stored_model_paths(models_path = "output"):
    return [os.path.join(models_path, dir) for dir in os.listdir(models_path)]