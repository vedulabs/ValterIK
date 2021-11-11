from utils.normalization import *
from termcolor import colored
import tensorflow as tf
import tensorflowjs as tfjs


class LeftSideIKSolver:
    __instance = None
    leftSideBoundaries = None
    leftSideFK = None
    targetPosInputTensor = None
    linksOutputTensor = None
    leftSideIKModel = None

    def __init__(self):
        if not LeftSideIKSolver.__instance:
            print('LeftSideIKSolver __init__ method called..')
        else:
            print('LeftSideIKSolver instance already created:', self.getInstance())

    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = LeftSideIKSolver()
        return cls.__instance

    @classmethod
    def setLeftSideBoundaries(cls, leftSideBoundariesjson):
        cls.leftSideBoundaries = leftSideBoundariesjson
        print('Left Side normalization boundaries:', cls.leftSideBoundaries)

    @classmethod
    def setLeftSideFK(cls, leftSideFKjson):
        cls.leftSideFK = leftSideFKjson
        print('Left Side FK set size:', len(cls.leftSideFK))

    @classmethod
    def prepareTrainingData(cls):
        if cls.leftSideFK is None:
            print(colored('Left Side FK Set is empty', 'yellow'))
            return None

        targetPosInput = list(map(lambda leftSideFKEntry: [
            normalizeNegPos(leftSideFKEntry[0],
                            cls.leftSideBoundaries['target']['x']['min'],
                            cls.leftSideBoundaries['target']['x']['max']),
            normalizeNegPos(leftSideFKEntry[1],
                            cls.leftSideBoundaries['target']['y']['min'],
                            cls.leftSideBoundaries['target']['y']['max']),
            normalizeNegPos(leftSideFKEntry[2],
                            cls.leftSideBoundaries['target']['z']['min'],
                            cls.leftSideBoundaries['target']['z']['max'])
        ], cls.leftSideFK))

        linksOutput = list(map(lambda leftSideFKEntry: [
            normalizeNegPos(leftSideFKEntry[3],
                            cls.leftSideBoundaries['links']['shoulderLeftLink']['min'],
                            cls.leftSideBoundaries['links']['shoulderLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[4],
                            cls.leftSideBoundaries['links']['limbLeftLink']['min'],
                            cls.leftSideBoundaries['links']['limbLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[5],
                            cls.leftSideBoundaries['links']['armLeftLink']['min'],
                            cls.leftSideBoundaries['links']['armLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[6],
                            cls.leftSideBoundaries['links']['forearmRollLeftLink']['min'],
                            cls.leftSideBoundaries['links']['forearmRollLeftLink']['max']),
        ], cls.leftSideFK))

        cls.targetPosInputTensor = tf.convert_to_tensor(targetPosInput, dtype=tf.float32)
        cls.linksOutputTensor = tf.convert_to_tensor(linksOutput, dtype=tf.float32)
        print(colored('targetPosInputTensor:\n', 'green'), cls.targetPosInputTensor)
        print(colored('linksOutputTensor:\n', 'green'), cls.linksOutputTensor)

    @classmethod
    def prepareTrainingDataWithPalmDirection(cls):
        if cls.leftSideFK is None:
            print(colored('Left Side FK Set is empty', 'yellow'))
            return None

        targetPosInput = list(map(lambda leftSideFKEntry: [
            normalizeNegPos(leftSideFKEntry[0],
                            cls.leftSideBoundaries['target']['x']['min'],
                            cls.leftSideBoundaries['target']['x']['max']),
            normalizeNegPos(leftSideFKEntry[1],
                            cls.leftSideBoundaries['target']['y']['min'],
                            cls.leftSideBoundaries['target']['y']['max']),
            normalizeNegPos(leftSideFKEntry[2],
                            cls.leftSideBoundaries['target']['z']['min'],
                            cls.leftSideBoundaries['target']['z']['max']),
            leftSideFKEntry[8],
            leftSideFKEntry[9],
            leftSideFKEntry[10]
        ], cls.leftSideFK))

        linksOutput = list(map(lambda leftSideFKEntry: [
            normalizeNegPos(leftSideFKEntry[3],
                            cls.leftSideBoundaries['links']['shoulderLeftLink']['min'],
                            cls.leftSideBoundaries['links']['shoulderLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[4],
                            cls.leftSideBoundaries['links']['limbLeftLink']['min'],
                            cls.leftSideBoundaries['links']['limbLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[5],
                            cls.leftSideBoundaries['links']['armLeftLink']['min'],
                            cls.leftSideBoundaries['links']['armLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[6],
                            cls.leftSideBoundaries['links']['forearmRollLeftLink']['min'],
                            cls.leftSideBoundaries['links']['forearmRollLeftLink']['max']),
            normalizeNegPos(leftSideFKEntry[7],
                            cls.leftSideBoundaries['links']['forearmLeftFrame']['min'],
                            cls.leftSideBoundaries['links']['forearmLeftFrame']['max']),
        ], cls.leftSideFK))

        cls.targetPosInputTensor = tf.convert_to_tensor(targetPosInput, dtype=tf.float32)
        cls.linksOutputTensor = tf.convert_to_tensor(linksOutput, dtype=tf.float32)
        print(colored('targetPosInputTensor:\n', 'green'), cls.targetPosInputTensor)
        print(colored('linksOutputTensor:\n', 'green'), cls.linksOutputTensor)

    @classmethod
    def prepareModel(cls):
        cls.leftSideIKModel = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_shape=[3], units=64, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        cls.leftSideIKModel.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.MSE,
            # loss=tf.keras.losses.MAE,
            metrics=['accuracy'])
        print(cls.leftSideIKModel.summary())

    @classmethod
    def prepareModelWithPalmDirection(cls):
        cls.leftSideIKModel = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_shape=[6], units=64, activation='relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(5, activation='linear')
        ])
        cls.leftSideIKModel.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            # loss=tf.keras.losses.MSE,
            loss=tf.keras.losses.MAE,
            metrics=['accuracy'])
        print(cls.leftSideIKModel.summary())

    @classmethod
    def fitModel(cls,
                 epochs=100,
                 batch_size=32,
                 verbose=1,
                 callbacks=None,
                 validation_split=0.15,
                 validation_data=None,
                 shuffle=True):
        cls.leftSideIKModel.fit(
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
        tfjs.converters.save_keras_model(cls.leftSideIKModel, './ikModels/leftSideIKModel')

