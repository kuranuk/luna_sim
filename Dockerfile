# Optional container image (LunarSim does not require Docker).
# Offline build: place a wheelhouse in ./wheelhouse/<tag> and pass --build-arg WHEELHOUSE=wheelhouse/<tag>
FROM python:3.12-slim
ARG WHEELHOUSE=""
WORKDIR /opt/lunarsim
COPY requirements.lock pyproject.toml README.md ./
COPY lunarsim ./lunarsim
COPY configs ./configs
COPY scripts ./scripts
COPY ${WHEELHOUSE:-requirements.lock} /tmp/wheelhouse
RUN if [ -d /tmp/wheelhouse ]; then \
        pip install --no-cache-dir --no-index --find-links /tmp/wheelhouse -r requirements.lock && \
        pip install --no-cache-dir --no-index --find-links /tmp/wheelhouse --no-deps . ; \
    else \
        pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . ; \
    fi && rm -rf /tmp/wheelhouse
RUN useradd --create-home lunarsim
USER lunarsim
ENTRYPOINT ["lunarsim"]
CMD ["--help"]
