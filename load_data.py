'''
Load data from text file stored in the following format:
  each line is formatted as:
    [number of pairs] [vocab word index]:[count] ...
  for example:
    3 2:10 5:5 7:10
'''

from scipy.sparse import dok_matrix
from collections import Counter

try:
    import tqdm
    _default_bar_type = 'terminal'
except ImportError:
    tqdm = None
    _default_bar_type = None

def load_data(filename, bar_type = _default_bar_type):
    with open(filename, 'r') as f:
        info = []
        vocab_size = 0
        for line in _prog_bar(bar_type, f.readlines(), desc='Reading lines'):
            xs = line.split()
            num_indices = int(xs[0])
            assert len(xs) == num_indices + 1
            count = Counter()
            for x in xs[1:]:
                i, n = map(int, x.split(':'))
                count[i] += n
                vocab_size = max(vocab_size, i+1)
            info.append(count)

    data = dok_matrix((len(info), vocab_size))

    for row_index, row in enumerate(_prog_bar(bar_type, info, desc='Filling matrix')):
        for i, n in row.items():
            data[row_index, i] = n

    return data.tocsr()

def _prog_bar(bar_type, iterable=None, **kwargs):
    if iterable is not None:
        kwargs["iterable"] = iterable

    if bar_type == 'notebook':
        return tqdm.tqdm_notebook(**kwargs)
    elif bar_type == 'terminal':
        return tqdm.tqdm(**kwargs)
    else:
        return iterable
