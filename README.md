We present reproduction instructions for the paper "Low-Dimensional manifolds support multiplexed integrations in Recurrent Neural Networks" by Arnaud Fanthomme and Rémi Monasson.


# Supported infrastructures
The following was tested using two different compute architectures:

- A Debian machine (#1 SMP Debian 4.9.130-2 (2018-10-27)) with CPU (Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz), and GPU
(Nvidia Tesla K40c, driver version 390.87).

- A Ubuntu macine with CPU (AMD Ryzen 7 1700x) and GPU (Nvidia 1070).

We did not test reproduction under MacOS or Windows operating systems, reproduction on those systems should still be possible but
the external dependencies might be harder to install.

Similarly, we do not encourage reproduction on CPU only machines as the runtimes might get unreasonably long, but there should
be no other major limitation.

# External dependencies
For installations of all relevant dependencies, we include in this archive a "conda_env_specs.txt" file.
After installing the Anaconda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment manager,
simply run the following:

conda create --name MYENV --file conda_env_specs.txt
conda activate MYENV

where MYENV should be replaced with an appropriate environment name.

After all installations succeed, you can move on to next step. You might be able to get away with changing some package
versions if the specified one is not available, but this might in turn require some changes to the code (in particular
  if having to use older versions of pytorch)

# Initial setup
After extracting this archive, please run setup.py to create the necessary folder for saves (out/), generate the test sequences and run some unittests.
It is expected behavior that this outputs a bunch of logging errors followed by logging infos confirming that these errors were expected.

After running these, use the launchers to run experiments, and aggregators to produce figures using results of several experiments.
