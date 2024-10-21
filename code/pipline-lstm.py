import os
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping


class LstmPipline():
    """
    This is a high-level overview of what the class does.

    More detailed description if necessary.
    """

    def __init__(self, X_train, X_test, y_train, y_test, model_dir='/home/mzero/main/repo/micro-ants/st.maarten_cpi_investigation/st.maarten_consumer_price_index_prediction/models'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.tuner = None
        self.best_model = None
        self.best_configs = None
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    
    def model_builder(self, hp):
        """
        Perform some action with param3.

        Args:
            param3 (type): Description of param3.

        Returns:
            return_type: Description of the return value.
        """


        model = Sequential()

        # Tuning the number of units in the first LSTM layer
        model.add(LSTM(units=hp.Int('units', 
                                    min_value=32, 
                                    max_value=128, 
                                    step=16), 
                    input_shape=(X_train.shape[1], X_train.shape[2])))

        # Tuning the dropout rate
        model.add(Dropout(rate=hp.Float('dropout', 
                                        min_value=0.0, 
                                        max_value=0.5, 
                                        step=0.1)))
        
        model.add(LSTM(units=hp.Int('units', 
                                    min_value=32, 
                                    max_value=128, 
                                    step=16)))

        model.add(De)

        # Output layer
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 
                                                                            min_value=1e-4, 
                                                                            max_value=1e-2, 
                                                                            sampling='log')),
                    loss=keras.losses.MeanSquaredError(),
                    metrics=[keras.metrics.MeanAbsoluteError()])

        
        return model


    def tuner_initiator(self):
        """
        Perform some action with param3.

        Args:
            param3 (type): Description of param3.

        Returns:
            return_type: Description of the return value.
        """

        self.tuner = kt.Hyperband(
            self.model_builder,
            objective='val_loss',
            max_epochs=50,
            factor=3,
            directory='/home/mzero/main/repo/micro-ants/st.maarten_cpi_investigation/st.maarten_consumer_price_index_prediction/tuner',
            project_name='Consumer Price Predictions'
        )


    def parameter_search(self):
        """
        Perform some action with param3.

        Args:
            param3 (type): Description of param3.

        Returns:
            return_type: Description of the return value.
        """

        self.tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.3, callbacks=[EarlyStopping(monitor='val_loss', patience=5)]) # Search for the best parameters 
        best_configs= self.tuner.get_best_hyperparameters(num_trials=1)[0]# Get the optimal hyperparameters

        return best_configs


    def output_optimimal_configs(self, best_configs):
        """
        Perform some action with param3.

        Args:
            param3 (type): Description of param3.

        Returns:
            return_type: Description of the return value.
        """

        units = best_configs.get('units')
        dropout = best_configs.get('dropout')
        learning_rate = best_configs.get('learning_rate')
        optimal_epochs = best_configs.get('tuner/epochs')

        print(f"""
        Optimal hyperparameters:
        - Units in LSTM layer: {units}
        - Dropout rate: {dropout}
        - Learning rate: {learning_rate}
        - Optimal number of epochs: {optimal_epochs} """)


    def evaluation_initiator(self):
        num_iterations = 10
        random_seeds = np.random.randint(0, 1000, size=num_iterations)

        f1_scores = []

        for seed in random_seeds:
            f1 = self.train_evaluate_model(self.best_configs,seed, X, y)
            f1_scores.append(f1)

        f1_scores = np.array(f1_scores)

        return f1_score


    def train_evaluate_model(self, best_configs, random_seed, X, y):
        """
        Perform some action with param3.

        Args:
            param3 (type): Description of param3.

        Returns:
            return_type: Description of the return value.
        """
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        optimal_epochs = best_configs.get('tuner/epochs')

        hypermodel = self.tuner.hypermodel.build(best_configs)
        hypermodel.fit(self.X_train, self.y_train, epochs=optimal_epochs, validation_split=0.3)
        y_pred = np.argmax(hypermodel.predict(X_test), axis=1)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        return f1

    
    def load_best_model(self):
        """
        Perform some action with param3.

        Args:
            param3 (type): Description of param3.

        Returns:
            return_type: Description of the return value.
        """
        self.best_model = tf.keras.models.load_model(os.path.join(self.model_dir, 'best_model.h5'))


    def predict(self, X_new):
        """
        Makes predictions on new data using the trained best model.

        Args:
            X_new (np.ndarray): New input data for prediction.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        if self.best_model is None:
            self.load_best_model()

        predictions = self.best_model.predict(X_new)
        
        return predictions


    def run(self):
        self.tuner_initiator()
        best_configs = self.parameter_search()
        self.output_optimimal_configs(best_configs)


tester = LstmPipline(X_train, X_test, y_train, y_test)
print(tester.run())