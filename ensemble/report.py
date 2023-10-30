from clearml import Task, OutputModel
import os

os.chdir("ensemble")

PROJECT_NAME = "YOLO"
TASK_NAME = "ensemble_reporting"
FRAMEWORK = "ensemble"
WEIGHTS_PATH = "ensemble"

task = Task.init(project_name = PROJECT_NAME,
                 task_name = TASK_NAME,
                 output_uri = True)
out_model = OutputModel(task = task, framework = FRAMEWORK)
out_model.update_weights_package(weights_path = f"{WEIGHTS_PATH}", auto_delete_file = False)
