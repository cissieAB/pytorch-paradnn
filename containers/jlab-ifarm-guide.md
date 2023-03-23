# Used `ifarm` singularity containers and conda environments

### Tensorflow

[Dockerfile.nv-tf](Dockerfile.nv-tf) is a container based on NVIDIA's ngc container
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

[jupyter-ai.yml](jupyter-ai.yml) is based on the conda env pulled from
`/scigroup/scicomp/jupyterhub/ai/ai-notebook.sif`. The container env has appropriate `mysql` package but not `cv2`.

```python
import cv2  # verify opencv
import MySQLdb  # verify mysql
```

I removed some error-triggering entries and
re-built the conda env with below command.

```bash
conda env create -n hydra-test --debug --force --file ai.yml
```
