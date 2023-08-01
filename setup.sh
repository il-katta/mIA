#!/bin/bash
set -xe
PYTHON_VERSION=3.10

cd "$(dirname "$0")"

if ! [ -d .venv-${PYTHON_VERSION} ] ; then
  python${PYTHON_VERSION} -m venv .venv-${PYTHON_VERSION}
fi

# shellcheck source=.venv-3.10/bin/activate
source ".venv-${PYTHON_VERSION}/bin/activate"

pip install -r requirements.txt

BNB_SOURCE_DIR="$(pwd)/.venv-${PYTHON_VERSION}/bitsandbytes"
rm -rf "${BNB_SOURCE_DIR}"
git clone https://github.com/timdettmers/bitsandbytes.git "${BNB_SOURCE_DIR}"
pushd "${BNB_SOURCE_DIR}"
  LD_LIBRARY_PATH=/opt/cuda/lib/ CUDA_HOME=/opt/cuda/ CUDA_VERSION=121 make cuda12x
  pip install .
popd
rm -rf "${BNB_SOURCE_DIR}"

BNB_CUDA_VERSION=121 LD_LIBRARY_PATH=/opt/cuda/lib/ python -m bitsandbytes

#pip install ninja
#MAX_JOBS=4 pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
