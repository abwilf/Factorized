import sys
import random
import argparse
from os.path import join
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir')
parser.add_argument('--a')
parser.add_argument('--b')
parser.add_argument('--c')

args = parser.parse_args()

print(args)

# print(sys.argv)
# if random.random() < .5:
#     sys.stderr.write("ERROR DESCRIPTION")
#     exit(1)
# else:
#     print('hi there!')

# with open(join(args.out_dir, 'success.txt'), 'w'):
#     pass


