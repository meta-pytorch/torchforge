# Running experiments on Slurm

First make sure you have followed the environment setup instructions in
https://github.com/meta-pytorch/torchforge/blob/main/README.md.

When running GRPO on Slurm, the "controller" (main script, grpo/main.py) can either run
locally on your login node, and launch the workers to Slurm; or the controller can
run on a remote node along with the workers.

To run with controller locally (good for debugging):
```
python -m apps.grpo.main --config experimental/slurm/qwen3_8b.yaml
```

To run with the controller remotely use the `submit.sh` script:

```
./experimental/slurm/submit.sh qwen3_8b
./experimental/slurm/submit.sh qwen3_32b
./experimental/slurm/submit.sh qwen3_30b_a3b
```

Note, the controller and workers will be launched in two different Slurm jobs.
