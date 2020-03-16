from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from keras.utils import plot_model
from keras.models import Model
import os

class Neural_Network:
    def __init__(
        self,
            feature_dim,
        num_classes=1,
        hidden_size=[],
        use_batch_norm=False,
        use_dropout=False,
    ):

        self.name = "fully_connected"
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.build_model()

    def build_model(self):
        input_fc = Input(shape=(self.feature_dim,), name="FC_input")
        fc_layer = input_fc

        for i in range(len(self.hidden_size)):
            fc_layer = Dense(units=self.hidden_size[i], name="Dense_layer_" + str(i))(fc_layer)
            if self.use_batch_norm:
                fc_layer = BatchNormalization(name= 'Batch_norm_' + str(i))(fc_layer)
            if self.use_dropout:
                fc_layer = Dropout(rate=0.5, name= 'Dropout_' + str(i))(fc_layer)

        fc_layer = Dense(units=self.num_classes, name="Dense_layer_last")(fc_layer)
        fc_layer = Dropout(rate=0.5, name= 'Dropout_last')(fc_layer)

        if self.num_classes == 1:
            fc_layer = Activation(activation="sigmoid", name="Activation_sigmoid")(fc_layer)
        else:
            fc_layer = Activation(activation="softmax", name="Activation_softmax")(fc_layer)

        input_model = input_fc
        output_model = fc_layer
        self.model_fc = Model(input_model, output_model)

    def compile(self, learning_rate=0.001, learning_decay=0.0001):
        self.learning_rate = learning_rate

        optimizer = Adam(learning_rate=learning_rate, decay=learning_decay)

        self.model_fc.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(
        self,
        x_train,
        y_train,
        batch_size=64,
        epochs=10,
        n_splits=3,
        seed=42,
        verbose=0,
    ):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(x_train, y_train.response):
            x_1 = x_train.iloc[train_idx, :]
            y_1 = y_train.iloc[train_idx, :].response.to_numpy()

            x_2 = x_train.iloc[test_idx, :]
            y_2 = y_train.iloc[test_idx, :].response.to_numpy()

            self.model_fc.fit(
                x_1,
                y_1,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_2, y_2),
                verbose=verbose,
            )

    def save_model(self, folder='runs/NN'):

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.plot_model(folder)
        self.model_fc.save(os.path.join(folder, "model_" + self.name + ".h5"))

    def plot_model(self, run_folder):
        plot_model(
            self.model_fc,
            to_file=os.path.join(run_folder, "model_" + self.name + ".png"),
            show_shapes=True,
            show_layer_names=True,
        )