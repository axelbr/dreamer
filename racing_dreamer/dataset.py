from rlephant import ReplayStorage
import tensorflow as tf

def load(filename: str, train: float):
    pass

def load_lidar(filename: str,  train: float, shuffle: bool = True):
    storage = ReplayStorage(filename=filename)
    transitions = sum([len(e) for e in storage])
    def generator():
        for episode in storage:
            for transition in episode:
                yield transition.observation['lidar']

    dataset = tf.data.Dataset.from_generator(generator, tf.float64)
    training_data_size = int(transitions * train)
    test_data_size = transitions - training_data_size
    train_data = dataset.take(training_data_size)
    test_data = dataset.skip(training_data_size).take(test_data_size)

    if shuffle:
        train_data = train_data.shuffle(training_data_size)
    return train_data, test_data