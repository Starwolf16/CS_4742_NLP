text = "aaaa bbb aaa bbb ababab acacac cacacad ccca dcdcdccdddccc cbbcbccb acac bdbdbd dbdbdb dadaaddadadddaaa ddd ccc bbb cdcdcdcd ccddcd dcdcdcdc"
text = text.replace(" ", "")
total_chars = len(text)

letter_counts = {
    'a' : text.count('a'),
    'b' : text.count('b'),
    'c' : text.count('c'),
    'd' : text.count('d')
}

print(f'Total Characters: {total_chars}\nLetter Counts: {letter_counts}')

probability_dict = {
    'a' : letter_counts["a"] / total_chars,
    'b' : letter_counts["b"] / total_chars,
    'c' : letter_counts["c"] / total_chars,
    'd' : letter_counts["d"] / total_chars
}

print(f'Probability of \'a\' occurring: {probability_dict["a"]}')
print(f'Probability of \'b\' occurring: {probability_dict["b"]}')
print(f'Probability of \'c\' occurring: {probability_dict["c"]}')
print(f'Probability of \'d\' occurring: {probability_dict["d"]}\n')

test_text = "aabcacddbcbbdaadda"
test_text_len = len(test_text)

test_text_prob = probability_dict['a']

for letter in test_text:
    test_text_prob = test_text_prob * probability_dict[letter]

perplexity = (1/test_text_prob) ** (1/test_text_len)
print(f'Perplexity: {perplexity}')