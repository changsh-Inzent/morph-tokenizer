import json
import argparse
import tqdm

MORPHEME_SEPARATOR = "▁"
EOJEOL_SEPARATOR = " "

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')

args = parser.parse_args()

total_sentences = 0
total_eojeols = 0

with open(args.output, 'w') as f:
    input_data = json.loads(open(args.input).read())

    for document_item in tqdm.tqdm(input_data['document']):
        for sentence_item in document_item['sentence']:
            total_sentences += len(sentence_item)

            if sentence_item['word'] and sentence_item['morpheme']:
                word_string = EOJEOL_SEPARATOR.join(word_item['form'] for word_item in sentence_item['word'])

                labels = []
                old_word_id = 0
                morphemes_per_eojeol = []
                morphemes = []
                for morpheme_item in sentence_item['morpheme']:
                    # 어절의 처음이 아닌 형태소 앞에 ▁를 붙여줍니다.
                    if morpheme_item['word_id'] != 1 and morpheme_item['word_id'] != old_word_id:
                        morphemes_per_eojeol.append(MORPHEME_SEPARATOR.join(morphemes))
                        morphemes = []
                        old_word_id = morpheme_item['word_id']

                    morphemes.append(morpheme_item['form'])

                    labels.append(morpheme_item['label'])

                morphemes_per_eojeol.append(MORPHEME_SEPARATOR.join(morphemes))

                morpheme_string = EOJEOL_SEPARATOR.join(morphemes_per_eojeol)

                total_eojeols += len(sentence_item['word'])

                f.write(json.dumps({'sentence': word_string, 'morphemes': morpheme_string, 'labels': labels}) + '\n')

print(f'Total sentences: {total_sentences}')
print(f'Total eojeols: {total_eojeols}')