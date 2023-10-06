from collections import Counter

input_nums = "112324567890104455668892100228899774410102997744333536373839300102030304050607"

digit_list = [int(digit) for digit in input_nums]
print(len(digit_list))

counts = dict(Counter(digit_list))

total = 0

for key, value in counts.items():
    print(f'{key}: {value}')
    total += value

print(total)