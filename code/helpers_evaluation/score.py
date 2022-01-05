

def print_score(output, ground_truth):
    score = 0
    max_score = 0
    misses = 0
    for i in range(len(ground_truth)):

        if ground_truth[i] == 'verified':
            max_score += 1
            if output[i] == 'verified':
                score += 1
        else:
            if output[i] == 'verified':
                score -= 2
                misses += 1

    print(f'\nScore: {score}/{max_score}')
    if misses:
        print(f'WARNING: UNSOUND, misclassified {misses} images as verified')


if __name__ == '__main__':
    GROUND_TRUTH = ['verified', 'verified', 'verified', 'verified', 'verified', 'verified', 'not verified', 'verified', 'verified', 'verified', 'verified', 'not verified', 'verified', 'not verified', 'not verified', 'verified', 'not verified', 'verified', 'verified', 'verified']

    with open('helpers_evaluation/results.txt') as f:
        lines = f.readlines()

    output = []
    for i in range(1, len(lines)):
        output.append(lines[i].strip())
    
    print_score(output, GROUND_TRUTH)
