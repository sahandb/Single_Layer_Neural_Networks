# imports
import numpy as np
import glob
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise

dirTrain = '../pic/TrainData/*.png'
dirTest = '../pic/TestData/*.png'
dirNoisy = '../pic/somepics/*.*'


def resizeImage(location, saveLoc):
    imgMatrixList = list()
    noisyImg = list()
    resizedImg = list()
    imageName = list()
    imageLabel = list()
    for filename in glob.glob(location):
        im = Image.open(filename).convert('L')  # read image and turn it to gray
        bw = im.point(lambda r: 0 if r < 120 else 255, '1')  # create bw !but i use it cuz its faster!
        # bw = np.asarray(im).copy()
        # # Pixel range is 0...255, 256/2 = 128 but  120 is better for (threshold) here
        # bw[bw < 110] = 0  # Black
        # bw[bw >= 110] = 255  # White
        # bww = Image.fromarray(bw)  # Now we put it back
        imr = bw.resize((622, 534))  # resize image to max one (here us 622 * 534) !!! hard code !!!
        noisyimage = Image.fromarray((255 * (random_noise(np.asarray(imr), mode='s&p', amount=0.15, salt_vs_pepper=0.25))).astype(np.uint8))
        noisyImg.append(np.array(noisyimage))
        resizedImg.append(imr)  # append resized images
        imageName.append(filename.split("\\")[1])  # save File names
        imageLabel.append(filename.split("\\")[1][6])
        iml = np.where(np.array(imr) == True, 1, -1)  # turn 0 to -1 (black) and 255 to 1 (white)
        imgMatrixList.append(iml)  # matrix of images
    for (i, new) in enumerate(resizedImg):  # recreate images with new size and labeled name
        new.save('{}{}'.format(saveLoc, imageName[i]))
    return np.array(noisyImg), np.array(imgMatrixList), np.array(imageLabel)


# read data and preprocess x and y
noisy_matrix_train, feature_train_matrix, target_train = resizeImage(dirTrain, '../pic/resized/train/SLDP/')
noisy_matrix_test, feature_test_matrix, target_test = resizeImage(dirTest, '../pic/resized/test/SLDP/')
noisy_matrix_somepics, feature_noisy_matrix, target_somepics = resizeImage(dirNoisy, '../pic/resized/somepics/SLDP/')

# fix labels for all data
target_train = np.where(np.array(target_train) == 'a', 0, target_train)
target_test = np.where(np.array(target_test) == '6', 3, target_test)
target_test = np.where(np.array(target_test) == 'd', 1, target_test)
target_test = np.where(np.array(target_test) == 'e', 0, target_test)
target_test = np.where(np.array(target_test) == 'v', 2, target_test)

# reformat matrix to vector for feeding perceptron
feature_train_vector = feature_train_matrix.reshape(
    (len(feature_train_matrix), feature_train_matrix.shape[1] * feature_train_matrix.shape[2]))
feature_test_vector = feature_test_matrix.reshape(
    (len(feature_test_matrix), feature_test_matrix.shape[1] * feature_test_matrix.shape[2]))
feature_noisy_vector = feature_noisy_matrix.reshape(
    (len(feature_noisy_matrix), feature_noisy_matrix.shape[1] * feature_noisy_matrix.shape[2]))

noisy_vector_train = noisy_matrix_train.reshape(
    (len(noisy_matrix_train), noisy_matrix_train.shape[1] * noisy_matrix_train.shape[2]))
noisy_vector_test = noisy_matrix_test.reshape(
    (len(noisy_matrix_test), noisy_matrix_test.shape[1] * noisy_matrix_test.shape[2]))
noisy_vector_somepics = noisy_matrix_somepics.reshape(
    (len(noisy_matrix_somepics), noisy_matrix_somepics.shape[1] * noisy_matrix_somepics.shape[2]))

# Split the train data into 70% training data and 30% test data
split_x_trainTRAIN, split_x_testTRAIN, split_y_trainTRAIN, split_y_testTRAIN = train_test_split(feature_train_vector,
                                                                                                target_train,
                                                                                                test_size=0.3)

# Create a perceptron object with the parameters: 60 iterations (epochs) over the data, and a learning rate of 0.15
ppn_split = Perceptron(max_iter=60, eta0=0.15, random_state=0)
ppn = Perceptron(max_iter=60, eta0=0.15, random_state=0)

# Train the perceptron
ppn_split.fit(split_x_trainTRAIN, split_y_trainTRAIN)
ppn.fit(feature_train_vector, target_train)

# Apply the trained perceptron on the features data to make predicts for the target test data
target_pred_trainTRAIN = ppn_split.predict(split_x_trainTRAIN)
target_pred_testTRAIN = ppn_split.predict(split_x_testTRAIN)

target_pred_train = ppn.predict(feature_train_vector)
target_pred_test = ppn.predict(feature_test_vector)
target_pred_noisy = ppn.predict(feature_noisy_vector)

noisy_target_pred_train = ppn.predict(noisy_vector_train)
noisy_target_pred_test = ppn.predict(noisy_vector_test)
noisy_target_pred_somepics = ppn.predict(noisy_vector_somepics)

# View model accuracy
# Defined as (1.0 - (# wrong predictions / # total observations))

# just test the train alone with split
print("Accuracy on just train data set with 30% test and 70% train \n")
print('Accuracy on train TRAIN: %.2f' % accuracy_score(split_y_trainTRAIN, target_pred_trainTRAIN))
print('Accuracy on test TRAIN: %.2f' % accuracy_score(split_y_testTRAIN, target_pred_testTRAIN))
# all data
print("\nAccuracy for all train and test")
print('Accuracy on train: %.2f' % accuracy_score(target_train, target_pred_train))
print('Accuracy on test: %.2f' % accuracy_score(target_test, target_pred_test))
print('Accuracy on noisy: %.2f' % accuracy_score(target_somepics, target_pred_noisy))
print("Accuracy for all train and test with noise")
print('Accuracy on train with noise: %.2f' % accuracy_score(target_train, noisy_target_pred_train))
print('Accuracy on test with noise: %.2f' % accuracy_score(target_test, noisy_target_pred_test))
print('Accuracy on somepics with noise: %.2f' % accuracy_score(target_somepics, noisy_target_pred_somepics))
