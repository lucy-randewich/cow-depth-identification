from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import shutil
import os
import numpy as np

# Get filepaths of all individual cow folders
cows = glob.glob('all-images/*')

# Make empty test and train folders
if not os.path.exists("images/train/"):
    os.mkdir("images/train")
    os.mkdir("images/test")

# Split each cow folder into train and test examples
cow_index = 1
for i, cow in enumerate(cows):
    cow_number = str(cow_index).rjust(3, '0')
    images = glob.glob(glob.escape(cow) + "/*")
    df = pd.DataFrame(images)

    # Grab 5 random images from cow for test, skip cow if less than 10 total images
    if df.shape[0] > 10:
        cow_index+=1
        drop_indices = np.random.choice(df.index, 5, replace=False)
        testing_data = df.iloc[drop_indices]
        training_data = df.drop(drop_indices)

        # Populate train and test folders
        for train_example in training_data[0]:
            # cow_number = str(int(os.path.dirname(train_example).rsplit('/', 1)[1]) + 1).rjust(3, '0')
            if not os.path.exists("images/train/" + cow_number):
                os.mkdir("images/train/" + cow_number)
            shutil.copyfile(train_example, "images/train/" + cow_number + "/" + os.path.basename(train_example))

        for test_example in testing_data[0]:
            # cow_number = str(int(os.path.dirname(test_example).rsplit('/', 1)[1]) + 1).rjust(3, '0')
            if not os.path.exists("images/test/" + cow_number):
                os.mkdir("images/test/" + cow_number)
            shutil.copyfile(test_example, "images/test/" + cow_number + "/" + os.path.basename(test_example))
