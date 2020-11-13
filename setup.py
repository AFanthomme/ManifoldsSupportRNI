### Do all necessary preprocessing
import os
import unittests

### For storing all results
os.makedirs('out', exist_ok=True)
os.makedirs('precomputed', exist_ok=True)
os.makedirs('precomputed/test_sequences', exist_ok=True)
os.makedirs('unittests', exist_ok=True)

### Generate test sequences
from datagen import TestSequencesPrecomputer
precomputer = TestSequencesPrecomputer()
precomputer._build_trajs()


### Some unittests
unittests.test_lstsq()
unittests.test_orth_torch()
unittests.test_sqrtm_torch()
unittests.plot_test_sequences()
unittests.test_data_sampler()
unittests.test_many_channels_net()
unittests.test_batch_loss()
