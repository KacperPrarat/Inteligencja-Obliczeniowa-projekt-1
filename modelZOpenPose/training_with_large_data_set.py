import keras.backend as K
import scipy.io
import numpy as np
import tensorflow as tf
import cv2
from keras import layers, Model
from keras.optimizers import Adam
from keras.applications import ResNet50


# Define the OpenPose model architecture
def OpenPoseModel():
    # Define the input layer
    inputs = layers.Input(shape=(image_height, image_width, num_channels))

    # Backbone network (e.g., VGG, ResNet)
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    # Additional convolutional layers for key point detection
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(resnet.output)
    # Add more convolutional layers as needed

    # Final output layers for key point detection
    keypoints = layers.Conv2D(num_keypoints, kernel_size=(1, 1), activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=keypoints)
    return model

# Load and preprocess training data


def load_and_preprocess_image(file_path, target_height, target_width):
    # Load image using OpenCV
    image = cv2.imread(file_path)
    
    # Resize image to target height and width
    image = cv2.resize(image, (target_width, target_height))
    
    # Convert image to float32 and normalize pixel values to range [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def load_data(mat_file_path, image_height, image_width):
    # Load .mat file
    data = scipy.io.loadmat(mat_file_path)

    # Extract annotations
    annolist = data['RELEASE']['annolist'][0][0]
    img_train = data['RELEASE']['img_train'][0][0]
    single_person = data['RELEASE']['single_person'][0][0]
    act = data['RELEASE']['act'][0][0]

    # Initialize lists to store data
    images = []
    bounding_boxes = []
    keypoints = []

    # Iterate through annotations
    for idx in range(len(annolist)):
        image_name = annolist[idx]['image']['name'][0][0][0]
        imgidx = int(image_name.split('.')[0]) - 1  # Extract image index
        is_training = img_train[0][imgidx]

        # Check if the image is for training
        if is_training:
            # Load and preprocess image
            image_path = 'path_to_your_images/' + image_name
            image = load_and_preprocess_image(image_path, image_height, image_width)
            
            # Extract body annotations for a person
            annorect = annolist[idx]['annorect']
            for ridx in range(len(annorect)):
                if not annorect[ridx].size:  # Skip empty annotations
                    continue
                body_ann = annorect[ridx]
                x1 = body_ann['x1'][0][0][0]
                y1 = body_ann['y1'][0][0][0]
                x2 = body_ann['x2'][0][0][0]
                y2 = body_ann['y2'][0][0][0]
                bbox = [x1, y1, x2, y2]
                bounding_boxes.append(bbox)

                # Extract key point annotations
                annopoints = body_ann['annopoints']['point'][0][0][0]
                key_points = []
                for point in annopoints:
                    x = point['x'][0][0][0]
                    y = point['y'][0][0][0]
                    is_visible = point['is_visible'][0][0][0]
                    key_points.append([x, y, is_visible])
                keypoints.append(key_points)

                images.append(image)

    return images, bounding_boxes, keypoints



# Define loss function for key point detection

def keypoint_loss(y_true, y_pred):
    # Calculate mean squared error (MSE) loss between ground truth (y_true) and predicted keypoints (y_pred)
    mse_loss = K.mean(K.square(y_true - y_pred), axis=-1)  # Compute MSE along the last axis (keypoint dimension)

    return mse_loss


# Train the model
def train_model(model, X_train, Y_train):
    # Compile the model
    model.compile(optimizer=Adam(), loss=keypoint_loss)

    # Train the model
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.2)
    return model

if __name__ == "__main__":
    # Parameters
    image_height = 256
    image_width = 256
    num_channels = 3
    num_keypoints = 17  # Number of key points (e.g., body joints)
    batch_size = 32
    num_epochs = 50

    # Load and preprocess training data
    X_train, Y_train = load_data("mpii_human_pose_v1_u12_1.mat")

    # Create and compile the model
    model = OpenPoseModel()

    # Train the model
    trained_model = train_model(model, X_train, Y_train)

    # Save the trained model
    trained_model.save("openpose_model.h5")
