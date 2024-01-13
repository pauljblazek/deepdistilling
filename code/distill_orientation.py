from distill import condense, test_code

def main():
    problem_name = 'orientation'
    ENN_layers_filenames = ['/data/orientation/ENN/orientation_1555919_1.csv',
                        '/data/orientation/ENN/orientation_1555919_2.csv',
                        '/data/orientation/ENN/orientation_1555919_3.csv']
    input_dims = [(28, 28)]

    print("Running distill_orientation.py")

    condense(problem_name, ENN_layers_filenames, input_dims)

    print("ENN successfully condensed. Distilled code is in output_orientation.py")

    directory = "/data/orientation"
    datasets = {'standard': ["train/Standard_images.csv", "train/Standard_labels.csv"],
                'diagonal': ["test/Diagonal_images.csv", "test/Diagonal_labels.csv"],
                'zigzag': ["test/Zigzag_images.csv", "test/Zigzag_labels.csv"],
                'dotted': ["test/Dotted_images.csv", "test/Dotted_labels.csv"],
                'outline': ["test/Outline_images.csv", "test/Outline_labels.csv"]
                }
    from output_orientation import ENN_orientation
    test_code(ENN_orientation, directory, datasets, input_dims[0])
    print()

main()
