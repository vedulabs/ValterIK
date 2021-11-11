import json
from leftSideIKSolver import LeftSideIKSolver

leftSideIKSolver = None


def load_data():
    print('Loading training data...')
    # with open('fkData/leftSideFK.json', 'r') as leftSideFKFile:
    # with open('fkData/leftSideFK_5k.json', 'r') as leftSideFKFile:
    with open('fkData/leftSideFK_withPalmPadDirection.json', 'r') as leftSideFKFile:
        leftSideFKjson = json.load(leftSideFKFile)
    # with open('fkData/leftSideNormBoundaries.json', 'r') as leftSideBoundariesFile:
    # with open('fkData/leftSideNormBoundaries_5k.json', 'r') as leftSideBoundariesFile:
    with open('fkData/leftSideNormBoundaries_withPalmPadDirection.json', 'r') as leftSideBoundariesFile:
        leftSideBoundariesjson = json.load(leftSideBoundariesFile)
    return leftSideFKjson, leftSideBoundariesjson


if __name__ == '__main__':
    leftSideFKjson, leftSideBoundariesjson = load_data()
    leftSideIKSolver = LeftSideIKSolver()
    leftSideIKSolver.setLeftSideBoundaries(leftSideBoundariesjson)
    leftSideIKSolver.setLeftSideFK(leftSideFKjson)

    # leftSideIKSolver.prepareTrainingData()
    # leftSideIKSolver.prepareModel()

    leftSideIKSolver.prepareTrainingDataWithPalmDirection()
    leftSideIKSolver.prepareModelWithPalmDirection()

    leftSideIKSolver.fitModel()
    leftSideIKSolver.saveModel()

