#sets the path to the directory containing training images
TrainingImagePath= './img'
# imports the ImageDataGenerator class from Keras, which is used for generating augmented image data for training the model
from keras.preprocessing.image import ImageDataGenerator
#instance of ImageDataGenerator is created for training data with specified augmentation parameters like shear range, zoom range, and horizontal flipping
train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
#An instance of ImageDataGenerator is created for test data without any augmentation
test_datagen = ImageDataGenerator()

#This generates a flow of augmented training images from the specified directory path. The images are resized to (64, 64) pixels, and the class mode is set to 'categorical' for multiclass classification
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#This generates a flow of test images from the same directory path as the training images. The images are resized to (64, 64) pixels, and the class mode is set to 'categorical'
test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
#This prints the mapping of class labels to their respective indices in the test set
test_set.class_indices
#This assigns the mapping of class labels to their respective indices in the training set to the TrainClasses variable
TrainClasses=training_set.class_indices

#This creates a mapping (ResultMap) between the class indices and their corresponding class labels
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
#This saves the ResultMap dictionary as a pickle file named "ResultsMap.pkl"
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)

#This prints the mapping of class indices to class labels
print("Mapping of Face and its ID",ResultMap)

#This calculates and prints the number of output neurons in the final layer of the neural network, which is equal to the number of unique classes
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)

#These lines import the required classes from Keras for building the convolutional neural network (CNN) architecture
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#This initializes a sequential model
classifier= Sequential()

#adds a convolutional layer with 32 filters of size 5x5, a stride of 1x1, and 'relu' activation function. The input shape of the layer is (64, 64, 3), indicating 64x64 RGB images
classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
#adds a max pooling layer with a pool size of 2x2
classifier.add(MaxPool2D(pool_size=(2,2)))
#adds another convolutional layer with 64 filters of size 5x5 and 'relu' activation function
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
#adds another max pooling layer with a pool size of 2x2
classifier.add(MaxPool2D(pool_size=(2,2)))
#flattens the output from the previous layer into a 1D vector
classifier.add(Flatten())
#adds a fully connected layer with 64 neurons and 'relu' activation function
classifier.add(Dense(64, activation='relu'))
#adds the output layer with a number of neurons equal to the number of unique classes. The activation function is 'softmax' for multiclass classification
classifier.add(Dense(OutputNeurons, activation='softmax'))
# compiles the model by specifying the loss function, optimizer, and metrics for evaluation
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
#imports the time module and records the current time as the starting time for training
import time
StartTime=time.time()
#fits the model to the training data using the fit_generator function. It trains the model for 50 epochs, with 1 step per epoch, and validates the model using the test data for each epoch
classifier.fit_generator(
                    training_set,
                    steps_per_epoch=1,
                    epochs=50,
                    validation_data=test_set,
                    validation_steps=1)
#calculates the total time taken for training and prints it in minutes
EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')



#This loads and preprocesses a test image from the specified path. The image is resized to (64, 64) pixels, converted to a NumPy array, and expanded to have an extra dimension to match the input shape of the model
import numpy
from keras.preprocessing import image
import tensorflow as tf

ImagePath='./p.jpg'
test_image=tf.keras.utils.load_img(ImagePath,target_size=(64, 64))
test_image=tf.keras.utils.img_to_array(test_image)

test_image=numpy.expand_dims(test_image,axis=0)
#This uses the trained model to predict the class probabilities for the test image
result=classifier.predict(test_image,verbose=0)

#This prints the predicted class label for the test image by finding the class index with the highest probability and mapping it to the corresponding class label using ResultMap
print('Prediction is: ',ResultMap[numpy.argmax(result)])