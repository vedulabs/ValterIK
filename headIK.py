import json
from headIKSolver import HeadIKSolver

headIKSolver = None


def load_data():
    print('Loading training data...')
    with open('fkData/headFK.json', 'r') as headFKFile:
        headFKjson = json.load(headFKFile)
    with open('fkData/headNormBoundaries.json', 'r') as headBoundariesFile:
        headBoundariesjson = json.load(headBoundariesFile)
    return headFKjson, headBoundariesjson


if __name__ == '__main__':
    # print(tf.version.VERSION)
    # tf.config.list_physical_devices('GPU')

    headFKjson, headBoundariesjson = load_data()
    headIKSolver = HeadIKSolver()
    headIKSolver.setHeadBoundaries(headBoundariesjson)
    headIKSolver.setHeadFK(headFKjson)
    headIKSolver.prepareTrainingData()
    headIKSolver.prepareModel()
    headIKSolver.fitModel()
    headIKSolver.saveModel()

