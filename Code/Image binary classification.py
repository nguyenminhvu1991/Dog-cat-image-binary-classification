"""
PROJECT: IMAGE CLASSIFICATION FOR DOG - CAT IMAGEs FROM KAGGLE
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy 
import cv2
import os
import glob

# input images
for img in glob.glob("D:/SP19/CIS-559_data mining/PROJECT/train1/*.jpg"): #folder train1 contains multiple dog and cat images in .jpg
    imagePaths = list(glob.glob("D:/SP19/CIS-559_data mining/PROJECT/train1/*.jpg"))

#Extract the image into vector
def image_vector(image, size=(128, 128)):
	return cv2.resize(image, size).flatten()

# initialize the pixel intensities matrix, labels list
imagemaxtrix = []
imagelabels = []

#Build image vector matrix
for (i, path) in enumerate(imagePaths):
	# load the image and extract the class label, image intensities
	image = cv2.imread(path)
	label = path.split(os.path.sep)[-1].split(".")[0]
	pixels = image_vector(image)

	# update the images and labels matricies respectively
	imagemaxtrix.append(pixels)	
	imagelabels.append(label)

imagemaxtrix = numpy.array(imagemaxtrix)
imagelabels = numpy.array(imagelabels)

#Prepare data for training and testing
(train_img, test_img, train_label, test_label) = train_test_split(
	imagemaxtrix, imagelabels, test_size=0.2, random_state=50)

'''SVM MODEL IN SKLEARN'''
model1 = SVC(max_iter=-1, kernel ='linear', class_weight='balanced', gamma ='scale')#kernel linear is better Gausian kernel here
model1.fit(train_img, train_label)
acc1 = model1.score(test_img, test_label)
print("SVM model accuracy: {:.2f}%".format(acc1 * 100))
#ww= model1.coef_
#print(ww)
#print(model1)


'''KNN MODEL IN SKLEARN'''
model2 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
model2.fit(train_img, train_label)
acc2 = model2.score(test_img, test_label)
print("KNN model accuracy: {:.2f}%".format(acc2 * 100))


'''ANN MODEL IN SKLEARN'''
model3 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
                      solver='sgd', tol=1e-4, random_state=1,
                      learning_rate_init=.1)
model3.fit(train_img, train_label)
acc3 = model3.score(test_img, test_label)
acc4 = model3.score(train_img, train_label)
print(" TEST accuracy: {:.2f}%".format(acc3 * 100))
print(" TRAIN accuracy: {:.2f}%".format(acc4 * 100))


'''ANN MODEL IN TENSORFLOW AND KERAS'''
#change the label from [cat,dog] into [0,1]
for index, i in enumerate(train_label):
     if i =='cat': train_label[index] = 1
     else: train_label[index] = 0
train_label    

for index, i in enumerate(test_label):
     if i =='cat': test_label[index] = 1
     else: test_label[index] = 0
test_label

from tensorflow.keras.models import Sequential
model = Sequential()
from tensorflow.keras.layers import Dense# Add the first hidden layer
model.add(Dense(50, activation='relu', input_dim=49152)) #Dense: many connect many
model.add(Dense(25, activation='relu'))# Add the second hidden layer
model.add(Dense(1, activation='sigmoid'))# Add the output layer
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #loss (internal the model) vs. accuracy (accuracy of prediction)
# Train the model for 200 epochs
model.fit(train_img, train_label, epochs=200) #train (80%) , test (15%), validation (5%)
scores = model.evaluate(train_img, train_label)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(test_img, test_label)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))



'''PREDICATION SAMPLE'''
rawImages1=[]
pixel1= image_vector(cv2.imread("unseen_image.jpg"))
rawImages1.append(pixels)	
rawImages1 = numpy.array(rawImages1)
prediction1 = model1.predict(rawImages1)
print(prediction1)
prediction2 = model2.predict(rawImages1)
print(prediction2)

