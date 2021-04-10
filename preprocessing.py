import os

import numpy as np

def _move_files(src_path, dst_path, fraction=0.2):
    files = os.listdir(src_path)
    files_to_mode = int(fraction * len(files))

    for file in np.random.choice(files, files_to_mode, False):
        src = os.path.join(src_path, file)
        dst = os.path.join(dst_path, file)
        os.replace(src, dst)


def extract_test(src_path, dst_path, split=0.2):
    for directory in os.listdir(src_path):        
        src = os.path.join(src_path, directory)
        dst = os.path.join(dst_path, directory)

        try:
            os.mkdir(dst)
        except FileExistsError:
            pass

        _move_files(src, dst, split)


def count_images_per_class(directory):

    counter = []

    for dir in os.listdir(directory):
        counter.append((dir, len(os.listdir(os.path.join(directory, dir)))))

    print(np.array(sorted(counter, key=lambda x : x[1], reverse=True)))
           
