"""
Sequence data object layer.

It's not clear that it's really a ``Sequence`` that we want, in the sense of
``collections.abc`` (https://docs.python.org/3/library/collections.abc.html).
Namely, we don't care about ``Reversible`` in most cases.
Perhaps we need an intermediate type?

"""

from collections.abc import Sequence


# No wrapper hooks have been added (yet)
# This was just to verify that ``Sequence`` mixins can resolve all methods from
# just __len__ and __getitem__ (but how, I don't know yet, seems it's burried in C)
class Seq(Sequence):
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item]
