import os
import sys
import shutil
from os import listdir
from os.path import isfile, join


source_dir = sys.argv[1]
destination_dir = sys.argv[2]

os.makedirs(destination_dir, exist_ok=True)

actor_folders = os.listdir(source_dir)

for actor_folder in actor_folders:
    actor_path = join(source_dir, actor_folder)

    if os.path.isdir(actor_path):
       
        for file_name in listdir(actor_path):
            file_path = join(actor_path, file_name)

            if isfile(file_path) and file_name.endswith(".wav"):
                
                emotion_code = file_name.split('-')[2]
       
                destination_folder = join(destination_dir, emotion_code)
                os.makedirs(destination_folder, exist_ok=True)

                shutil.move(file_path, join(destination_folder, file_name))
                
                print(f"Moved {file_name} to {destination_folder}")
