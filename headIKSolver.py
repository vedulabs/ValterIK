from utils.normalization import *
from termcolor import colored
import tensorflow as tf
import tensorflowjs as tfjs


class HeadIKSolver:
    __instance = None
    headBoundaries = None
    headFK = None
    targetPosInputTensor = None
    linksOutputTensor = None
    headIKModel = None

    def __init__(self):
        if not HeadIKSolver.__instance:
            print('HeadIKSolver __init__ method called..')
        else:
            print('HeadIKSolver instance already created:', self.getInstance())

    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = HeadIKSolver()
        return cls.__instance

    @classmethod
    def setHeadBoundaries(cls, headFKBoundariesjson):
        cls.headBoundaries = headFKBoundariesjson
        print('Head normalization boundaries:', cls.headBoundaries)

    @classmethod
    def setHeadFK(cls, headFKjson):
        cls.headFK = headFKjson
        print('Head FK set size:', len(cls.headFK))

    @classmethod
    def prepareTrainingData(cls):
        if cls.headFK is None:
            print(colored('Head FK Set is empty', 'yellow'))
            return None

        targetPosInput = list(map(lambda headFKEntry: [
            normalizeNegPos(headFKEntry[0],
                            cls.headBoundaries['target']['x']['min'],
                            cls.headBoundaries['target']['x']['max']),
            normalizeNegPos(headFKEntry[1],
                            cls.headBoundaries['target']['y']['min'],
                            cls.headBoundaries['target']['y']['max']),
            normalizeNegPos(headFKEntry[2],
                            cls.headBoundaries['target']['z']['min'],
                            cls.headBoundaries['target']['z']['max'])
        ], cls.headFK))

        linksOutput = list(map(lambda headFKEntry: [
            normalizeNegPos(headFKEntry[3],
                            cls.headBoundaries['links']['headYawLink']['min'],
                            cls.headBoundaries['links']['headYawLink']['max']),
            normalizeNegPos(headFKEntry[4],
                            cls.headBoundaries['links']['headTiltLink']['min'],
                            cls.headBoundaries['links']['headTiltLink']['max'])
        ], cls.headFK))

        cls.targetPosInputTensor = tf.convert_to_tensor(targetPosInput, dtype=tf.float32)
        cls.linksOutputTensor = tf.convert_to_tensor(linksOutput, dtype=tf.float32)
        print(colored('targetPosInputTensor:\n', 'green'), cls.targetPosInputTensor)
        print(colored('linksOutputTensor:\n', 'green'), cls.linksOutputTensor)

    @classmethod
    def prepareModel(cls):
        cls.headIKModel = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_shape=[3], units=32, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        cls.headIKModel.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.MSE,
            metrics=['accuracy'])
        print(cls.headIKModel.summary())

    @classmethod
    def fitModel(cls,
                 epochs=40,
                 batch_size=32,
                 verbose=1,
                 callbacks=None,
                 validation_split=0.15,
                 validation_data=None,
                 shuffle=True):
        cls.headIKModel.fit(
            cls.targetPosInputTensor,
            cls.linksOutputTensor,
            batch_size,
            epochs,
            verbose,
            callbacks,
            validation_split,
            validation_data,
            shuffle
        )


    @classmethod
    def saveModel(cls):
        tfjs.converters.save_keras_model(cls.headIKModel, './ikModels/headIKModel/')

