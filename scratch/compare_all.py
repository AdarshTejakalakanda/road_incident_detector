import json
post = json.load(open('artifacts/post_ab_full_benchmark_sweep.json'))
print('keys:', list(post.keys()))
print('overall_true_positives:', post.get('overall_true_positives'))
print('overall_false_negatives:', post.get('overall_false_negatives'))
print('overall_false_positives:', post.get('overall_false_positives'))
print('recall;', post.get('incident_recall'))
print('precision:', post.get('precision'))
print('f1:', post.get('f1_score'))
