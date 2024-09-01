import os
import random
import argparse

def create_test_file(image_dir, output_file):
    # Get all image files in the directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
    
    # Write the file paths to the test.txt file
    with open(output_file, 'w') as f:
        for image_file in image_files:
            f.write(image_file + '\n')
    
    print(f"Created {output_file} with {len(image_files)} image paths.")

def split_train_test(image_dir, eval_mode=False):
    # Read the image paths from the test.txt file
    test_file_path = os.path.join(image_dir, 'test.txt')
    with open(test_file_path, 'r') as f:
        image_paths = f.read().splitlines()
    
    # Shuffle the image paths to ensure randomness
    random.shuffle(image_paths)
    
    # Split into train and test sets (80% train, 20% test)
    split_index = int(0.8 * len(image_paths))
    train_set = image_paths[:split_index]
    test_set = image_paths[split_index:]
    
    if eval_mode:
        # Write the test set paths to a test_eval.txt file
        with open(os.path.join(image_dir, 'test_eval.txt'), 'w') as f:
            for path in test_set:
                f.write(path + '\n')
        print(f"Evaluation mode: Created test_eval.txt with {len(test_set)} image paths.")
    else:
        # Write the train set paths to a train.txt file
        with open(os.path.join(image_dir, 'train.txt'), 'w') as f:
            for path in train_set:
                f.write(path + '\n')
        # Write the test set paths to a test_split.txt file
        with open(os.path.join(image_dir, 'test_split.txt'), 'w') as f:
            for path in test_set:
                f.write(path + '\n')
        print(f"Training mode: Created train.txt with {len(train_set)} image paths and test_split.txt with {len(test_set)} image paths.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and split image paths for training and testing.")
    parser.add_argument('image_dir', type=str, help='Directory containing the images.')
    parser.add_argument('--eval', action='store_true', help='If passed, split into test set only.')
    
    args = parser.parse_args()
    
    # Step 1: Create the test.txt file
    create_test_file(args.image_dir, os.path.join(args.image_dir, 'test.txt'))
    
    # Step 2: Split into train/test sets if needed
    split_train_test(args.image_dir, eval_mode=args.eval)
