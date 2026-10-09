import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    post = json.load(f)
with open('artifacts/t_full_benchmark_sweep.json') as f:
    t = json.load(f)

print('Post-AB overall FPs:', post.get('overall_false_positives'))
print('T-full overall FPs:', t.get('overall_false_positives'))


for p1, t1 in zip(post['per_video_results'], t['per_video_results']):
    fp_p = p1.get('fp_confirmed_count', 0)
    fp_t = t1.get('fp_confirmed_count', 0)
    if fp_p != fp_t or fp_p > 0:
        print(f"{p1['video_id']} ({p1['split_type']}): Post-AB FP = {fp_p}, T FP = {fp_t}")
