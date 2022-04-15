import argparse
import predictor
import sys


def main(args):
    ni_ip = args.N
    #print('Nearest Ingredients Input: ', ni_ip)

    ingredients = []
    for item in args.ingredient:
        ingredients.append(item)

    output = predictor.predict_cuisine(ni_ip, ingredients)
    sys.stdout.write(output)


# Argument Parser Setup
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUISINE PREDICTOR')

    parser.add_argument('--N', type=int, help='Number of Nearest Ingredients', required=True)
    parser.add_argument('--ingredient', type=str, help='Ingredients in the dish', required=True, action='append')

    args = parser.parse_args()
    main(args)
