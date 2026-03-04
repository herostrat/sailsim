FROM python:3.11-slim

# Build dependencies for pypilot (SWIG, build tools, python headers, cmake for RTIMU)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential swig libffi-dev python3-dev cmake && \
    rm -rf /var/lib/apt/lists/*

# Pre-install Python dependencies that pypilot needs
RUN pip install --no-cache-dir \
    numpy scipy pywavefront gevent flask ujson pyserial pillow

# Build and install RTIMULib (required for pypilot's IMU module)
RUN git clone --depth=1 https://github.com/RTIMULib/RTIMULib2.git /tmp/rtimu && \
    cd /tmp/rtimu/Linux/python && \
    python3 setup.py install && \
    rm -rf /tmp/rtimu

# Clone pypilot
RUN git clone --depth=1 https://github.com/pypilot/pypilot.git /opt/pypilot
WORKDIR /opt/pypilot
RUN mkdir -p deps && python3 setup.py install
RUN rm -rf /opt/pypilot/pypilot

# Apply simulation patch: GPS-track fallback when IMU is unavailable
COPY apply_patch.py /tmp/apply_patch.py
RUN python3 /tmp/apply_patch.py && rm /tmp/apply_patch.py

# Copy simulation config
COPY pypilot_sim.conf /root/.pypilot/pypilot.conf

WORKDIR /root
EXPOSE 23322 20220
CMD ["pypilot"]
