# measurement plugin (scratch): every hypothesis test runs only its explicit @examples, no generation.
# the cost of a run that generates nothing = the fixed part of any run (collection, plain tests, @examples)
from hypothesis import Phase, settings
settings.register_profile('explicit_only', phases=[Phase.explicit], deadline=None)
settings.load_profile('explicit_only')
