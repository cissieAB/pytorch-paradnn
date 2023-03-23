# xmei@jlab.org Mar/22/2023

# Did not succeed in
# singularity pull --docker-login *.sif docker://ngc_container
# on ifarm (as guided at https://docs.nvidia.com/ngc/gpu-cloud/ngc-catalog-user-guide/index.html#converting-to-singularity-image)
# Thus rebuild the ngc container and push to my own docker account.
#
# Build docker image locally
# docker login -u xxmei https://index.docker.io/v2/
# docker push docker.io/xmei/nv-tf:tag
#
# On ifarm, pull it via
# singularity pull nv-tf2.sif docker://xmei/nv-tf:tag
# singularity remote login --username xxmei docker://docker.io

FROM nvcr.io/nvidia/tensorflow:22.09-tf2-py3
