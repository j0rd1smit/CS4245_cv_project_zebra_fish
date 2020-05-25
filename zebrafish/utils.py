import os

def get_all_stored_model_paths(models_path = "output"):
    return [os.path.join(models_path, dir) for dir in os.listdir(models_path)]


def is_in_main_dir():
    return os.path.basename(os.path.normpath(os.getcwd())) == "CS4245_cv_project_zebra_fish"