from score import print_score

GROUND_TRUTH = ['not verified', 'verified', 'verified', 'not verified', 'verified', 'not verified', 'verified', 'verified', 'not verified', 'verified', 'verified', 'verified', 'verified', 'verified', 'not verified', 'verified', 'verified', 'not verified', 'not verified', 'verified', 'verified', 'verified', 'verified', 'verified', 'verified']

with open('helpers_evaluation/results_prelim.txt') as f:
    lines = f.readlines()

output = []
for i in range(1, len(lines)):
    output.append(lines[i].strip())

print_score(output, GROUND_TRUTH)
