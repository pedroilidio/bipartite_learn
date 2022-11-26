import pstats, cProfile
from test_splitters import test_pbct_splitter_gso
from test_nd_classes import test_simple_tree_1d2d
from make_examples import make_interaction_regression

test_pbct_splitter_gso(verbose=True)
# cProfile.runctx("test_pbct_splitter_gso(verbose=True)", globals(), locals(), "Profile.prof")
# cProfile.runctx("test_simple_tree_1d2d(msl=1)", globals(), locals(), "Profile.prof")
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()