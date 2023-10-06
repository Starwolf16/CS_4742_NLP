from collections import Counter

sentence_1 = "a a a a b b b a a b c b c d a b a d b d d c d b "
sentence_2 = "b b b c c c d d c d c b c a c b c b c a c a b a d"
sentence_3 = "d b d b d b d c d a d a d c d d d b d c"
sentence_4 = "c d c b c b c a c c a c b c d c a a a b"
sentence_5 = "c d c d d d b b a a b a d"

sentence_1 = list(filter(lambda x : x != " ", sentence_1))
sentence_2 = list(filter(lambda x : x != " ", sentence_2))
sentence_3 = list(filter(lambda x : x != " ", sentence_3))
sentence_4 = list(filter(lambda x : x != " ", sentence_4))
sentence_5 = list(filter(lambda x : x != " ", sentence_5))

counter = 1
for item in [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]:
    
    total = 0
    counts = Counter(item)
    counts = list(counts)
    counts.sort()
    counts = dict(counts)
    print(f'Sentence: {counter}')
    for key, value in counts.items():
        print(f'{key}: {value}')
        total += value
    print(f'Total chars: {total}\n')
    counter += 1