import tensorflow as tf

class DQN:

    def __init__(self, model, loss='mse', learning_rate=1e-3, optimizer=tf.keras.optimizers.Adam):
        self.model = model
        self.model.compile(optimizer=optimizer(learning_rate), loss=loss)

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def load_model(self, model_file):
        self.model.load_weights(model_file)

    def save_model(self, model_file):
        self.model.save_weights(model_file)

    def copy_from(self, dqn):
        self.model.set_weights(dqn.get_model().get_weights())

    def create_target_network(self):
        return DQN(tf.keras.models.clone_model(self.model))

