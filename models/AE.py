from keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
    Conv2DTranspose,
    Dropout,
    ReLU,
    Activation,
    Lambda,
    BatchNormalization,
    Reshape
)
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.optimizers import Adadelta, Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.models import Model


import numpy as np
import os
import pickle
import json

# from utils.callbacks import CustomCallback, step_decay_schedule


class Autoencoder:
    def __init__(
        self,
        input_dim,
        hidden_size,
        feature_dim,
        use_batch_norm=False,
        use_dropout=False,
    ):
        self.name = "autoencoder"
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.build_model()

    def build_model(self):

        input_encoder = Input(shape=(self.input_dim,), name="encoder_input")
        encoded = input_encoder

        for i in range(len(self.hidden_size)):
            dense_layer = Dense(units=self.hidden_size[i], activation='relu', name="Dense_layer_" + str(i))
            encoded = dense_layer(encoded)
            if self.use_batch_norm:
                encoded = BatchNormalization(name= 'Batch_norm_' + str(i))(encoded)
            if self.use_dropout:
                encoded = Dropout(rate=0.25)(encoded)

        encoded = Dense(self.feature_dim, name='Dense_encoder_output')(encoded)

        if self.use_batch_norm:
            encoded = BatchNormalization()(encoded)
        if self.use_dropout:
            encoded = Dropout(rate=0.25)(encoded)

        encoder_output = encoded
        self.model_encoded = Model(input_encoder, encoder_output)

        # Decoder

        input_decoder = Input(shape=(self.feature_dim,), name="Decoder_input")
        decoded = input_decoder

        hidden_size_decoder = self.hidden_size
        for j in reversed(range(len(hidden_size_decoder))):
            dense_layer = Dense(units=hidden_size_decoder[j], activation='relu', name= 'Dense_decode_' + str(j))
            decoded = dense_layer(decoded)
            if self.use_batch_norm:
                decoded = BatchNormalization(name= 'Batch_norm_' + str(j))(decoded)
            if self.use_dropout:
                decoded = Dropout(rate=0.25, name= 'Dropout_' + str(j))(decoded)

        decoded = Dense(self.input_dim, name='Decoded_output', activation='sigmoid')(decoded)

        if self.use_batch_norm:
            decoded = BatchNormalization(name= 'Batch_norm_last')(decoded)
        if self.use_dropout:
            decoded = Dropout(rate=0.25, name= 'Dropout_last')(decoded)

        output_decoder = decoded
        self.model_decoded = Model(input_decoder, output_decoder)

        # Full AE
        input_model = input_encoder
        output_model = self.model_decoded(encoder_output)

        self.model_autoencoder = Model(input_model, output_model)

    def compile(self, learning_rate, learning_decay):
        self.learning_rate = learning_rate

        optimizer = Adam(learning_rate=learning_rate, decay=learning_decay)
        # def r_loss(y_true, y_pred):
        #     return K.mean(K.square(y_true - y_pred), axis = [1,2,3])

        self.model_autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')

    def load_weights(self, filepath):
        self.model_autoencoder.load_weights(filepath)

    def train(self, x_train, y_train,  batch_size, epochs, n_splits=5, seed=42, verbose=0):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(x_train, y_train.response):
            x_tr = x_train.iloc[train_idx, :]
            x_ts = x_train.iloc[test_idx, :]

            self.model_autoencoder.fit(x_tr, x_tr,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=(x_ts, x_ts),
                                       verbose=verbose)

    def save_model(self, folder='runs/AE'):

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.plot_model(folder)
        self.model_autoencoder.save(os.path.join(folder, "model_ae_" + self.name + ".h5"))
        self.model_encoded.save(os.path.join(folder, "model_encoded_" + self.name + ".h5"))

    def plot_model(self, run_folder):
        plot_model(
            self.model_autoencoder,
            to_file=os.path.join(run_folder, "model_" + self.name + ".png"),
            show_shapes=True,
            show_layer_names=True,
        )
