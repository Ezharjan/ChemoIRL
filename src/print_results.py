import json

# Load results
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*70)
print("ALL BASELINE METHODS - REAL EXPERIMENTAL RESULTS")
print("="*70)

print("\n{:<15} {:>10} {:>10} {:>10} {:>10}".format("Method", "Simple", "Complex", "Overall", "RRE"))
print("-" * 70)

# Sort by overall F1 score
sorted_methods = sorted(data.items(), key=lambda x: x[1]['overall'], reverse=True)

for method, scores in sorted_methods:
    rre_str = f"{scores['rre']:.3f}" if scores['rre'] is not None else "N/A"
    print("{:<15} {:>10.3f} {:>10.3f} {:>10.3f} {:>10}".format(
        method, 
        scores['simple'], 
        scores['complex'], 
        scores['overall'],
        rre_str
    ))

print("="*70)
print("\nTop methods by Overall F1:")
for method, scores in sorted_methods[:5]:
    print(f"- {method}: {scores['overall']:.3f}")

irl_methods = [
    (m, s) for m, s in sorted_methods
    if s.get('rre') is not None
]

if irl_methods:
    lowest_rre_method, lowest_rre_scores = min(irl_methods, key=lambda item: item[1]['rre'])
    print(f"\nLowest RRE among reward-learning methods: {lowest_rre_method} ({lowest_rre_scores['rre']:.3f})")

print("="*70)
