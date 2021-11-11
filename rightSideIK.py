import json
import tensorflow as tf
from tensorflow.python.client import device_lib
from rightSideIKSolver import RightSideIKSolver

rightSideIKSolver = None


def load_data():
    print('Loading training data...')
    # with open('fkData/rightSideFK.json', 'r') as rightSideFKFile:
    # with open('fkData/rightSideFK_1k.json', 'r') as rightSideFKFile:
    # with open('fkData/rightSideFK_5k.json', 'r') as rightSideFKFile:
    with open('fkData/rightSideFK_withPalmPadDirection.json', 'r') as rightSideFKFile:
        rightSideFKjson = json.load(rightSideFKFile)
    # with open('fkData/rightSideNormBoundaries.json', 'r') as rightSideBoundariesFile:
    # with open('fkData/rightSideNormBoundaries_1k.json', 'r') as rightSideBoundariesFile:
    # with open('fkData/rightSideNormBoundaries_5k.json', 'r') as rightSideBoundariesFile:
    with open('fkData/rightSideNormBoundaries_withPalmPadDirection.json', 'r') as rightSideBoundariesFile:
        rightSideBoundariesjson = json.load(rightSideBoundariesFile)
    return rightSideFKjson, rightSideBoundariesjson


if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')
    print(device_lib.list_local_devices())

    rightSideFKjson, rightSideBoundariesjson = load_data()
    rightSideIKSolver = RightSideIKSolver()
    rightSideIKSolver.setRightSideBoundaries(rightSideBoundariesjson)
    rightSideIKSolver.setRightSideFK(rightSideFKjson)

    # rightSideIKSolver.prepareTrainingData()
    # rightSideIKSolver.prepareModel()

    rightSideIKSolver.prepareTrainingDataWithPalmDirection()
    rightSideIKSolver.prepareModelWithPalmDirection()

    rightSideIKSolver.fitModel()
    rightSideIKSolver.saveModel()

