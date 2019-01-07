# Build features from scratch
Run following commands:

```bash
step01_prepare_input.py
step02_std_features.py
step03_redshift_features.py
step04_template_features.py
step05_misc_features.py
step06_compile_features.py
step07_train_lgbm.py
```

**IMPORTANT: It takes 6~9 months with 64 core machine in step04_template_features.py.** If you want to create features on your own,
it is highly recommended to split step05 into subsets and run each script on different machines. Increasing the number of CPUs didn't
improve performance (Parallelization in iminuit which is the backend of sncosmo.lc_fit doesn't scale well.
Data parallelism should be applied in this case, but iminuit and my implementation aren't...)

# Train nyanp's model from pre-compiled feature binaries

```bash
step07_train_lgbm.py
```

# Train yuval's or mamas's model from pre-compiled feature binaries

Copy feature binaries in share/ and follow the instruction in yuval's or mamas's document.