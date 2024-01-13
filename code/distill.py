from load_dd import translate_enn
import pandas as pd
import numpy as np
import os

def condense(problem_name, ENN_layers_filenames, input_dims, layer_names=None, return_style='max', logic_inputs=False):
    
    if layer_names is None:
        layer_names = ['I', 'D', 'S', 'C']

    translate_enn(ENN_layers_filenames, layer_names, input_dims, problem_name, return_style=return_style, logic_inputs=logic_inputs)

def test_code(func, directory, datasets, input_shape):
    def test(dataset_name, image_path, labels_path):    

        max_to_test = 5000

        df_img = pd.read_csv(image_path, header=None)
        df_label = pd.read_csv(labels_path, header=None)
        m = len(df_img)

        if m > max_to_test:
            indices = np.random.choice(m, max_to_test, replace=False)
        else:
            indices = np.arange(m)

        acc = 0
        for index in indices:
            input = np.reshape(np.array(df_img.iloc[index, :]), input_shape).astype(float)
            label = np.array(df_label.iloc[index])[0]
            result = func(input)
            if result == label:
                acc += 1
        acc /= len(indices)
        print("Accuracy on {} dataset: {}".format(
            dataset_name, round(acc,3)))


    for key in datasets.keys():
        image_path, label_path = datasets[key]
        image_path = os.path.join(directory, image_path)
        label_path = os.path.join(directory, label_path)
        test(key, image_path, label_path)
