import os
import shutil

logs_path = "logs_shelter"
if os.path.exists(logs_path):
    shutil.rmtree("logs_shelter")
