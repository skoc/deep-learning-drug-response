import yaml
import io


class Hyperparams:
    def __init__(self, file_parameters):
        with io.open(file_parameters) as f:
            hp_list = yaml.load(f, Loader=yaml.FullLoader)

        hp_ae = "hyperparameters_ae"
        hp_nn = "hyperparameters_nn"
        hp_train = "training"

        # Autoencoder Hyperparams
        self.learning_rate_ae = hp_list[hp_ae]["learning_rate"]
        self.learning_decay_ae = hp_list[hp_ae]["learning_decay"]
        self.hidden_sizes_ae = hp_list[hp_ae]["hidden_sizes"]
        self.drop_rate_ae = hp_list[hp_ae]["drop_rate"]
        self.feature_dim = hp_list[hp_ae]["feature_dim"]
        self.use_batch_norm_ae = hp_list[hp_ae]["use_batch_norm"]
        self.use_dropout_ae = hp_list[hp_ae]["use_dropout"]

        # Neural Network Hyperparams
        self.learning_rate_nn = hp_list[hp_nn]["learning_rate"]
        self.learning_decay_nn = hp_list[hp_nn]["learning_decay"]
        self.num_classes = hp_list[hp_nn]["num_classes"]
        self.hidden_sizes_nn = hp_list[hp_nn]["hidden_sizes"]
        self.use_batch_norm_nn = hp_list[hp_nn]["use_batch_norm"]
        self.use_dropout_nn = hp_list[hp_nn]["use_dropout"]

        # Training Hyperparams
        self.num_epochs_ae = hp_list[hp_train]["num_epochs_ae"]
        self.num_epochs_nn = hp_list[hp_train]["num_epochs_nn"]
        self.batch_size_ae = hp_list[hp_train]["batch_size_ae"]
        self.batch_size_nn = hp_list[hp_train]["batch_size_nn"]
        self.n_folds = hp_list[hp_train]["n_folds"]
        self.drug_type = hp_list[hp_train]["drug_type"]
        self.test_size = hp_list[hp_train]["test_size"]
        self.omics = hp_list[hp_train]["omics"]

        self.load_ae = hp_list[hp_train]["load_ae"]
        self.load_nn = hp_list[hp_train]["load_nn"]
