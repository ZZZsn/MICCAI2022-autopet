FROM pytorch/pytorch

FROM python:3.8-slim

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY --chown=algorithm:algorithm checkpoints/ /opt/algorithm/checkpoints/

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN python -m pip install --user -rrequirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm UNet.py /opt/algorithm/
COPY --chown=algorithm:algorithm test_loader.py /opt/algorithm/

ENTRYPOINT python -m process $0 $@
