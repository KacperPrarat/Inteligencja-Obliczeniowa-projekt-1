import os
# Specify paths
# annotations_path = './first_490_anotations.txt'
annotations_path = './last_210_anotations.txt'
# output_directory = './data/labels/train'
output_directory = './data/labels/val'

# Open the annotations file
with open(annotations_path, 'r') as f:
    # Read each line of the file
    for i, line in enumerate(f):
        # Create filename based on line number
        filename = os.path.join(output_directory, f'CCTV_{i + 491}.txt')
        # Write the line to the new file
        with open(filename, 'w') as new_file:
            new_file.write(line)
