import itertools

from scripts import nqa_utils
import tqdm

def calc_f1(raw_examples, answers):
    total_score = nqa_utils.Score()

    for example in raw_examples:
        if example['example_id'] not in answers:
            continue

        la = answers[example['example_id']]['long_answer']
        sa = answers[example['example_id']]['short_answer']

        long_pred = ''
        if la:
            long_pred = '{}:{}'.format(la[0], la[1])
        total_score.increment(long_pred, nqa_utils.long_annotations(example), [])

        short_pred = ''
        if sa:
            short_pred = '{}:{}'.format(sa[0], sa[1])
        total_score.increment(short_pred, nqa_utils.short_annotations(example), [])

    return total_score.F1()


def grid_search_weights(raw_examples, preds, candidates_dict,
                        features, examples, id_to_example, weight_ranges,
                        invalid_input_ids=[], pre_answers=None):
    combinations = list(itertools.product(*[weight_ranges[k] for k in weight_ranges.keys()]))
    results = {}
    for weight_vals in tqdm.tqdm(combinations):
        tmp_weights = {}
        
        weight_sum = 0
        threshold = 0
        for weight_name, weight_val in zip(weight_ranges.keys(), weight_vals):
                tmp_weights[weight_name] = weight_val
                if 'weight' in weight_name:
                    weight_sum += weight_val
                elif weight_name == 'conf_threshold':
                    threshold = weight_val

        if weight_sum < threshold or weight_sum > threshold * 2:
            continue
                
        tmp_answers = nqa_utils.compute_answers(preds, 
                                                candidates_dict, 
                                                features, 
                                                examples, 
                                                id_to_example,
                                                weights=tmp_weights,
                                                invalid_input_ids=invalid_input_ids)
        
        if pre_answers:
            tmp_answers = dict(list(pre_answers.items()) + list(tmp_answers.items()))
        
        f1 = calc_f1(raw_examples, tmp_answers)
        results[weight_vals] = f1
        
    sr = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return list(sr)

def print_scores(scores):
    print('Total Score', '\n', scores[0], '\n')
    print('Long Answer Score', '\n', scores[1], '\n')
    print('Short Answer Score', '\n', scores[2], '\n')