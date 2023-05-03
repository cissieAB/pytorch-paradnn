# `ifarm` singularity containers and conda environments

### Tensorflow

[Dockerfile.nv-tf](../containers/Dockerfile.nv-tf) is a container based on NVIDIA's ngc container
[nvcr.io/nvidia/tensorflow:22.09-tf2-py3](nvcr.io/nvidia/tensorflow:22.09-tf2-py3).
As I did not find the way to `singularity pull` directly from ngc, I relocated it to my own
dockerhub repo. Remember to reset the singularity cache and tmp dir before pull, otherwise
you will encounter "No space left on device" error.

```bash
# Build tf container on ifarm
# It's necessary to reset singularity cache and tmp dirs
export SINGULARITY_CACHEDIR=/work/epsci/xmei/singularity_
export SINGULARITY_TMPDIR=/work/epsci/xmei/singularity_tmp

singularity pull --disable-cache nv-tf.sif docker://xxmei/nv-tf:22.09
```

### Hydra test

The singularity containers for
[JLAB's jupyterhub](https://jupyterhub.jlab.org/hub/spawn) are at `/scigroup/scicomp/jupyterhub`.
`/scigroup/scicomp/jupyterhub/ai/ai-notebook.sif` is the one for "AI" group.
It has appropriate `mysql`/`cv2` python packages.

```python
# In python3
import cv2  # verify opencv, this is only valid on GPU node
import MySQLdb  # verify mysql
```

Launch this container

```bash
cd /group/halld/hydra/
srun --gres gpu:A100:2 --mem-per-cpu 4000 --cpus-per-task 4 -p gpu --pty bash
singularity shell --nv /scigroup/scicomp/jupyterhub/ai/ai-notebook.sif

python3 hydra_train.py -c ../Hydra.cfg -D CDC_occupancyChunks -e 1000 -g 2 -d
```

Training code is updated to TF2.11.1.
