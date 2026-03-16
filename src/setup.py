### Do all necessary preprocessing
import os
import tests_setup

### For storing all results. 
# These are rather heavy (few GB) but they are rather slow to regenerate 
# using current implementation
os.makedirs('out/precomputed_test_sequences', exist_ok=True)

### Generate test sequences
from datagen import TestSequencesPrecomputer
precomputer = TestSequencesPrecomputer()
precomputer._build_trajs()


### Testing some numerical routines:
# This one failed hard on GPU due to changed SVD behavior, but correct when taking sign ambiguityinto accounts
# Also, not actually used in any of the code...
tests_setup.test_orth_torch() 
tests_setup.test_lstsq()
tests_setup.test_sqrtm_torch()
tests_setup.test_data_sampler()
tests_setup.test_many_channels_net()
tests_setup.test_batch_loss()

print('\n'*3)
print('-'*40)
print('All tests ran successfully !')
print('-'*40)
print('\n'*3)