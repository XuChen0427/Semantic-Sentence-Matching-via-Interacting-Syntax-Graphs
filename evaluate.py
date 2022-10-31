import sys
from srcV3.evaluator import Evaluator


def main():
    argv = sys.argv
    if len(argv) == 2:
        model_path = argv[1:]
        evaluator = Evaluator(model_path)
        evaluator.evaluate()
        #evaluator.DecodeTrain()
        #evaluator.GetParsingMatrix()
    else:
        print('Usage: "python evaluate.py $model_path "')


if __name__ == '__main__':
    main()
