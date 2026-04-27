from Classificaltion_Evaluation import ClassificationEvaluation
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def Model_LSTM(trainX, trainY, testX, testy, BS=None, sol=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [5, 5]

    print('Model LSTM')
    IMG_SIZE = [1, 100]
    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((testX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testX.shape[0]):
        Test_Temp[i, :] = np.resize(testX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    model = Sequential()
    classes = trainY.shape[-1]
    model.add(LSTM(int(sol[0]), input_shape=(Train_X.shape[1], Train_X.shape[-1])))  # hidden neuron count(5 - 255)
    model.add(Dense(50, activation="relu"))
    model.add(Dense(classes, activation="relu"))  # activation="relu"
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  # mean_squared_error binary_crossentropy
    model.fit(Train_X, trainY, epochs=int(sol[1]), steps_per_epoch=10, batch_size=BS, verbose=2, validation_data=(Test_X, testy))
    testPredict = model.predict(Test_X)
    pred = np.asarray(testPredict).astype('int')
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = ClassificationEvaluation(testy, pred)
    return Eval, pred

