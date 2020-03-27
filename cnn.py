from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3),activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))


classifier.add(Flatten())

classifier.add(Dense(units = 128,activation = 'relu'))

classifier.add(Dense(units = 128,activation = 'relu'))

classifier.add(Dense(units = 3,activation = 'softmax'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('G:\Shreyas\Deep Learning\Projects\CNN\Friends\Categories',
                                                 target_size = (64,64),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory('G:\Shreyas\Deep Learning\Projects\CNN\Friends\Categories',
                                                 target_size = (64,64),
                                                 batch_size = 32)

classifier.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 1000)


classifier.save('model.h5')
