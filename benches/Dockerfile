# Image benches/plot.py runs in, built and run by bench.sh
FROM python:3.12-slim

RUN pip install --no-cache-dir matplotlib==3.9.2

# The container runs as the calling user, who has no home directory there.
# The font cache is built once here and readable by everyone.
ENV MPLCONFIGDIR=/opt/matplotlib
RUN python -c "import matplotlib.pyplot" && chmod -R a+rwX /opt/matplotlib
