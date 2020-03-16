import pandas as pd
import numpy as np

from models.AE import Autoencoder
from models.NN import Neural_Network
from data.data_prep import get_data
from utils.hyperparameters import Hyperparams

from sklearn.metrics import roc_auc_score
from keras.models import load_model


def train(X_train, y_train, hp):

    # Autoencoder
    if not hp.load_ae:
        input_dim = X_train.shape[1]
        autoencoder = Autoencoder(
            input_dim=input_dim,
            hidden_size=hp.hidden_sizes_ae,
            feature_dim=hp.feature_dim,
            use_batch_norm=hp.use_batch_norm_ae,
            use_dropout=hp.use_dropout_ae,
        )
        autoencoder.build_model()
        autoencoder.compile(learning_rate=hp.learning_rate_ae,
                            learning_decay=hp.learning_decay_ae)
        print("Training Autoencoder...")
        autoencoder.train(
            x_train=X_train,
            y_train=y_train,
            batch_size=hp.batch_size_ae,
            epochs=hp.num_epochs_ae,
            verbose=0
        )
        print("Trained Autoencoder...")
        autoencoder.save_model()

        encoder_model = autoencoder.model_encoded


    else:
        print("Loading Autoencoder Model...")
        encoder_model = load_model("runs/AE/model_encoded_autoencoder.h5")
        print("Loaded Autoencoder Model...")

    # Neural Network
    input_encoded = pd.DataFrame(encoder_model.predict(X_train))

    if not hp.load_nn:
        neural_net = Neural_Network(feature_dim=hp.feature_dim,
                                    num_classes=hp.num_classes)
        neural_net.compile(learning_rate=hp.learning_rate_nn,
                           learning_decay=hp.learning_decay_nn)
    else:
        print("Loading Fully Connected Model...")
        autoencoder = load_model("runs/NN/model_nn.h5")
        print("Loaded Fully Connected Model...")


    print("Training Neural Network...")
    neural_net.train(x_train=input_encoded,
                     y_train=y_train,
                     epochs=hp.num_epochs_nn,
                     n_splits=hp.n_folds,
                     batch_size=hp.batch_size_nn,
                     verbose=1
    )
    neural_net.save_model()
    print("Trained Neural Network...")

    return {"Encoder": encoder_model, "NN": neural_net.model_fc}


def main():
    # Hyperparameters
    SETUP_PATH = "config/setup.yml"
    hyperparameters = Hyperparams(SETUP_PATH)
    print(''.join("%s:\t%s\n" % item for item in vars(hyperparameters).items()))

    # Data
    print("Loading data...")
    X_train, y_train, X_test, y_test = get_data(drug=hyperparameters.drug_type,
                                                test_size=hyperparameters.test_size)
    print("Loaded data...")

    # Training
    models = train(X_train, y_train, hp=hyperparameters)

    # Testing
    pred  = models['NN'].predict(models['Encoder'].predict(X_test))
    print(pred)
    auc_score = roc_auc_score(y_test.response, np.squeeze(pred))
    print(f"\n--\nAUC Score: {auc_score:.3f}\n--\n")

if __name__ == "__main__":
    main()
