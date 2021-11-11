from utils.normalization import *
from termcolor import colored
import tensorflow as tf
import tensorflowjs as tfjs


class RightSideIKSolver:
    __instance = None
    rightSideBoundaries = None
    rightSideFK = None
    targetPosInputTensor = None
    linksOutputTensor = None
    rightSideIKModel = None

    def __init__(self):
        if not RightSideIKSolver.__instance:
            print('RightSideIKSolver __init__ method called..')
        else:
            print('RightSideIKSolver instance already created:', self.getInstance())

    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = RightSideIKSolver()
        return cls.__instance

    @classmethod
    def setRightSideBoundaries(cls, rightSideBoundariesjson):
        cls.rightSideBoundaries = rightSideBoundariesjson
        print('Right Side normalization boundaries:', cls.rightSideBoundaries)

    @classmethod
    def setRightSideFK(cls, rightSideFKjson):
        cls.rightSideFK = rightSideFKjson
        print('Right Side FK set size:', len(cls.rightSideFK))

    @classmethod
    def prepareTrainingData(cls):
        if cls.rightSideFK is None:
            print(colored('Right Side FK Set is empty', 'yellow'))
            return None

        targetPosInput = list(map(lambda rightSideFKEntry: [
            normalizeNegPos(rightSideFKEntry[0],
                            cls.rightSideBoundaries['target']['x']['min'],
                            cls.rightSideBoundaries['target']['x']['max']),
            normalizeNegPos(rightSideFKEntry[1],
                            cls.rightSideBoundaries['target']['y']['min'],
                            cls.rightSideBoundaries['target']['y']['max']),
            normalizeNegPos(rightSideFKEntry[2],
                            cls.rightSideBoundaries['target']['z']['min'],
                            cls.rightSideBoundaries['target']['z']['max'])
        ], cls.rightSideFK))

        linksOutput = list(map(lambda rightSideFKEntry: [
            normalizeNegPos(rightSideFKEntry[3],
                            cls.rightSideBoundaries['links']['shoulderRightLink']['min'],
                            cls.rightSideBoundaries['links']['shoulderRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[4],
                            cls.rightSideBoundaries['links']['limbRightLink']['min'],
                            cls.rightSideBoundaries['links']['limbRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[5],
                            cls.rightSideBoundaries['links']['armRightLink']['min'],
                            cls.rightSideBoundaries['links']['armRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[6],
                            cls.rightSideBoundaries['links']['forearmRollRightLink']['min'],
                            cls.rightSideBoundaries['links']['forearmRollRightLink']['max']),
        ], cls.rightSideFK))

        cls.targetPosInputTensor = tf.convert_to_tensor(targetPosInput, dtype=tf.float32)
        cls.linksOutputTensor = tf.convert_to_tensor(linksOutput, dtype=tf.float32)
        print(colored('targetPosInputTensor:\n', 'green'), cls.targetPosInputTensor)
        print(colored('linksOutputTensor:\n', 'green'), cls.linksOutputTensor)

    @classmethod
    def prepareTrainingDataWithPalmDirection(cls):
        if cls.rightSideFK is None:
            print(colored('Right Side FK Set is empty', 'yellow'))
            return None

        targetPosInput = list(map(lambda rightSideFKEntry: [
            normalizeNegPos(rightSideFKEntry[0],
                            cls.rightSideBoundaries['target']['x']['min'],
                            cls.rightSideBoundaries['target']['x']['max']),
            normalizeNegPos(rightSideFKEntry[1],
                            cls.rightSideBoundaries['target']['y']['min'],
                            cls.rightSideBoundaries['target']['y']['max']),
            normalizeNegPos(rightSideFKEntry[2],
                            cls.rightSideBoundaries['target']['z']['min'],
                            cls.rightSideBoundaries['target']['z']['max']),
            rightSideFKEntry[8],
            rightSideFKEntry[9],
            rightSideFKEntry[10]
        ], cls.rightSideFK))

        linksOutput = list(map(lambda rightSideFKEntry: [
            normalizeNegPos(rightSideFKEntry[3],
                            cls.rightSideBoundaries['links']['shoulderRightLink']['min'],
                            cls.rightSideBoundaries['links']['shoulderRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[4],
                            cls.rightSideBoundaries['links']['limbRightLink']['min'],
                            cls.rightSideBoundaries['links']['limbRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[5],
                            cls.rightSideBoundaries['links']['armRightLink']['min'],
                            cls.rightSideBoundaries['links']['armRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[6],
                            cls.rightSideBoundaries['links']['forearmRollRightLink']['min'],
                            cls.rightSideBoundaries['links']['forearmRollRightLink']['max']),
            normalizeNegPos(rightSideFKEntry[7],
                            cls.rightSideBoundaries['links']['forearmRightFrame']['min'],
                            cls.rightSideBoundaries['links']['forearmRightFrame']['max']),
        ], cls.rightSideFK))

        cls.targetPosInputTensor = tf.convert_to_tensor(targetPosInput, dtype=tf.float32)
        cls.linksOutputTensor = tf.convert_to_tensor(linksOutput, dtype=tf.float32)
        print(colored('targetPosInputTensor:\n', 'green'), cls.targetPosInputTensor)
        print(colored('linksOutputTensor:\n', 'green'), cls.linksOutputTensor)

    @classmethod
    def prepareModel(cls):
        cls.rightSideIKModel = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_shape=[3], units=64, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        cls.rightSideIKModel.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            # loss=tf.keras.losses.MSE,
            loss=tf.keras.losses.MAE,
            metrics=['accuracy'])
        print(cls.rightSideIKModel.summary())

    @classmethod
    def prepareModelWithPalmDirection(cls):
        cls.rightSideIKModel = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_shape=[6], units=64, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='linear')
        ])
        cls.rightSideIKModel.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            # loss=tf.keras.losses.MSE,
            loss=tf.keras.losses.MAE,
            metrics=['accuracy'])
        print(cls.rightSideIKModel.summary())

    @classmethod
    def fitModel(cls,
                 epochs=100,
                 batch_size=32,
                 verbose=1,
                 callbacks=None,
                 validation_split=0.15,
                 validation_data=None,
                 shuffle=True):
        cls.rightSideIKModel.fit(
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
        tfjs.converters.save_keras_model(cls.rightSideIKModel, './ikModels/rightSideIKModel')

