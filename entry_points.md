# Build features and models from scratch
Run following commands:

```bash
step01_prepare_input.py
step02_template_features.py
step03_std_features.py
step04_time_series_features.py
step05_redshift_features.py
step06_misc_features.py
step07_run_model.py
step08_make_shared_features.py
```

**IMPORTANT: It takes 6~9 months with single 64 core machine in step02_template_features.py.** If you want to create features on your own,
it is highly recommended to split step05 into subsets and run each script on different machines. Increasing the number of CPUs didn't
improve performance (Parallelization in iminuit which is the backend of sncosmo.lc_fit doesn't scale well.
Data parallelism should be applied in this case, but iminuit and my implementation aren't...)

To reproduce template features within a reasonable time, please follow the steps below:

1. Setup a GCP instance
1. Run `step01_prepare_input.py` in the instance
1. Create GCP image from the instance
1. Make 30 clones from the image
1. run `step02_template_features.py n` for each clone instance (n denotes index from 0 to 29)
1. gather all partial feature files into the master instance
1. run `step02_template_features.py merge` in the master instance

In this case, you don't need high-performance instance (the number of cores doesn't matter). 
8 CPU, 30 GB RAM instance is enough for these steps.

# Train nyanp's model from pre-compiled feature binaries

```bash
step07_run_model.py
```

# Train Yuval's or Mamas's model from pre-compiled feature binaries

Copy feature binaries in share/ and follow the instruction in Yuval's or Mamas's document.
