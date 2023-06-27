import sys
import random

lines = []

for line in sys.stdin:
    lines.append(line.strip())

random.shuffle(lines)

for line in lines:
    print(line)