from nltk import word_tokenize

def generate_skip_bigrams(tokens, skip=2):
    skip_bigrams = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + skip + 1, len(tokens))):
            skip_bigram = " ".join(tokens[i:j])
            skip_bigrams.append(skip_bigram)
    return skip_bigrams

def rouge_s(reference, candidate, skip=2):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)

    reference_skip_bigrams = generate_skip_bigrams(reference_tokens, skip)
    candidate_skip_bigrams = generate_skip_bigrams(candidate_tokens, skip)

    common_skip_bigrams = set(reference_skip_bigrams).intersection(candidate_skip_bigrams)
    
    if len(reference_skip_bigrams) == 0:
        precision = 0
    else:
        precision = len(common_skip_bigrams) / len(reference_skip_bigrams)

    return precision

system_summary = "Water spinach is a leaf vegetable commonly eaten in tropical areas of Asia"
human_summaries = [
    "Water spinach is a green leafy vegetable grown in the tropics",
    "Water spinach is a semi-aquatic tropical plant grown as a vegetable",
    "Water spinach is a commonly eaten leaf vegetable of Asia"
]

total_precision = 0

for human_summary in human_summaries:
    s = rouge_s(human_summary, system_summary, skip=2)
    total_precision += s

overall_rouge_s = total_precision / len(human_summaries)

print(f"Overall ROUGE-S Score: {overall_rouge_s:.4f}")
