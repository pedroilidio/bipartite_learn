import numpy as np
from hypertrees.model_selection import train_test_split_nd
from make_examples import make_interaction_data 
from sklearn.utils.validation import _num_samples

def stratify(y, bins=10):
    ax_means = y.mean(axis=1), y.mean(axis=0)
    # cuts = [np.linspace(m.shape[0], bins) for m in ax_means)]
    cuts = [np.quantile(m, np.linspace(0., 1., bins)) for m in ax_means]
    encoded_y = [np.digitize(m, c) for m, c in zip(ax_means, cuts)] 
    return encoded_y 

X, y, _ = make_interaction_data((100, 100), (10, 10), random_state=9)
strat = stratify(y)
print('y mean', y.mean())

stratify_labels = stratify(y)
print(_num_samples(stratify_labels[0]))

split = train_test_split_nd((X, y), stratify=stratify_labels, random_state=0)
for k,v in split.items():
    print(k, 'shape:', v[1].shape, 'mean:', v[1].mean())
