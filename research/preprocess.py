import json
import random

from composer import decompose


MORPHEME_SEPARATOR = "▁"


def read_jsonl(src):
    with open(src) as f:
        for x in f:
            yield json.loads(x)

def write_jsonl(dest, data):
    with open(dest, 'w') as f:
        for x in data:
            f.write(json.dumps(x, separators=(',', ':'), ensure_ascii=False) + '\n')

def morph(src):
    for x in read_jsonl(src):
        s = ' '.join([decompose(w[0]) for w in x[1]])
        d = ' '.join([MORPHEME_SEPARATOR.join([decompose(t[0]) for t in l[1]]) for l in x[1]])
        yield (s, d)

if __name__ == '__main__':
    src = list(morph('raw.jsonl'))
    random.shuffle(src)

    left = int(len(src)*0.05)
    write_jsonl('test.jsonl', src[:left])
    write_jsonl('train.jsonl', src[left:])
