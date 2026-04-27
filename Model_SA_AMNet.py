from Classificaltion_Evaluation import ClassificationEvaluation
from Model_LSTM import Model_LSTM
from stellargraph.layer import GraphConvolution
from tensorflow.keras.layers import Input, Dropout, Dense, Concatenate
from tensorflow.keras.models import Model
from keras import backend as K
import keras
import numpy as np
from sklearn.model_selection import train_test_split


def multi_scale_gcn(n_nodes, n_features_1, n_features_2, n_features_3, n_classes, sol):
    kernel_initializer = "glorot_uniform"
    bias_initializer = "zeros"

    # Input for 1st scale (Feature Set I)
    x_features_1 = Input(shape=(n_nodes, n_features_1))
    x_adjacency_1 = Input(shape=(n_nodes, n_nodes))

    # Input for 2nd scale (Feature Set II)
    x_features_2 = Input(shape=(n_nodes, n_features_2))
    x_adjacency_2 = Input(shape=(n_nodes, n_nodes))

    # Input for 3rd scale (Feature Set III)
    x_features_3 = Input(shape=(n_nodes, n_features_3))
    x_adjacency_3 = Input(shape=(n_nodes, n_nodes))

    # First scale GCN
    x1 = Dropout(0.5)(x_features_1)
    x1 = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)([x1, x_adjacency_1])
    x1 = Dropout(0.5)(x1)
    x1 = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)([x1, x_adjacency_1])

    # Second scale GCN
    x2 = Dropout(0.5)(x_features_2)
    x2 = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)([x2, x_adjacency_2])
    x2 = Dropout(0.5)(x2)
    x2 = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)([x2, x_adjacency_2])

    # Third scale GCN
    x3 = Dropout(0.5)(x_features_3)
    x3 = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)([x3, x_adjacency_3])
    x3 = Dropout(0.5)(x3)
    x3 = GraphConvolution(32, activation='relu', use_bias=True, kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer)([x3, x_adjacency_3])

    # Concatenate the outputs from the three scales
    concatenated = Concatenate()([x1, x2, x3])

    # Classification layer
    x = Dense(int(sol[0]), activation='relu')(concatenated)
    x = Dense(n_classes, activation='sigmoid')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)  # Ensure the output shape is (None, n_classes)

    model = Model(inputs=[x_features_1, x_adjacency_1, x_features_2, x_adjacency_2, x_features_3, x_adjacency_3],
                  outputs=x)
    model.summary()

    return model


def create_adjacency_and_indices(data, n_nodes):
    n_samples = data.shape[0]
    adjacency_matrices = np.ones((n_samples, n_nodes, n_nodes)) - np.eye(n_nodes)
    indices = np.tile(np.arange(n_nodes), (n_samples, 1))
    return adjacency_matrices, indices


def Model_MultiScale_GCN_Feat(Feat__X1, Feat__X2, Feat__X3, Y, BS=None, sol=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [5, 5]
    n_nodes = 10
    n_features_1 = 100
    n_features_2 = 100
    n_features_3 = 100
    n_classes = Y.shape[-1]

    X1 = np.zeros((Feat__X1.shape[0], n_nodes, n_features_1))
    X2 = np.zeros((Feat__X2.shape[0], n_nodes, n_features_2))
    X3 = np.zeros((Feat__X3.shape[0], n_nodes, n_features_3))

    for i in range(Feat__X1.shape[0]):
        X1[i] = np.reshape(np.resize(Feat__X1[i], (n_nodes, n_features_1)), (n_nodes, n_features_1))
        X2[i] = np.reshape(np.resize(Feat__X2[i], (n_nodes, n_features_2)), (n_nodes, n_features_2))
        X3[i] = np.reshape(np.resize(Feat__X3[i], (n_nodes, n_features_3)), (n_nodes, n_features_3))

    Train_X1, Test_X1, Train_X2, Test_X2, Train_X3, Test_X3, Train_Y, Test_Y = train_test_split(X1, X2, X3, Y,
                                                                                                random_state=104,
                                                                                                test_size=0.25,
                                                                                                shuffle=True)

    train_adjacency, _ = create_adjacency_and_indices(Train_X1, n_nodes)
    test_adjacency, _ = create_adjacency_and_indices(Test_X1, n_nodes)

    # Create the multi-scale GCN model
    model = multi_scale_gcn(n_nodes, n_features_1, n_features_2, n_features_3, n_classes, sol)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([Train_X1, train_adjacency, Train_X2, train_adjacency, Train_X3, train_adjacency], Train_Y,
              epochs=int(sol[1]), batch_size=BS, steps_per_epoch=10,
              validation_data=([Test_X1, test_adjacency, Test_X2, test_adjacency, Test_X3, test_adjacency], Test_Y))

    # Predict and evaluate
    pred = model.predict([Test_X1, test_adjacency, Test_X2, test_adjacency, Test_X3, test_adjacency])
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    inp = model.input
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 3
    data = np.concatenate((X1, X2, X3), axis=1)

    Feats = []
    for i in range(data.shape[0]):
        print(i, data.shape[0])
        test = data[i, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()  # [func([test]) for func in functors]
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (data.shape[0], Feats.shape[-1]))
    return Feature


def Model_SA_AMNet(Images, BERT, Spectral, Target, sol=None, BS=None):
    if sol is None:
        sol = [5, 5, 5, 5]
    Feature = Model_MultiScale_GCN_Feat(Images, BERT, Spectral, Target, BS, sol[:2])
    train_X, test_X, train_Y, test_Y = train_test_split(Feature, Target, random_state=104, test_size=0.25, shuffle=True)
    Eval, Pred = Model_LSTM(train_X, train_Y, test_X, test_Y, BS, sol[2:])
    return Eval, Pred


