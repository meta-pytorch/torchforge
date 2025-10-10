python -m pip install --pre torch==2.9.0.dev20250905 --no-cache-dir --index-url https://download.pytorch.org/whl/nightly/cu129
python -m pip install -r .github/packaging/vllm_reqs.txt
python -m pip install vllm==0.10.1.dev0+g6d8d0a24c.d20251009.cu129 --no-cache-dir --index-url https://download.pytorch.org/whl/preview/forge
python -m pip install -r https://raw.githubusercontent.com/meta-pytorch/monarch/main/requirements.txt
python -m pip install torchmonarch --extra-index-url https://download.pytorch.org/whl/preview/forge
python -m pip install git+ssh://git@github.com/pytorch/torchtitan.git
python -m pip install git+ssh://git@github.com/meta-pytorch/torchstore.git
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/libpython3.10.so.1.0:${LD_LIBRARY_PATH}
export LD_PRELOAD=${CONDA_PREFIX}/lib/libpython3.10.so.1.0:${LD_PRELOAD}
pip install -e ".[dev]"
pytest tests/unit_tests
