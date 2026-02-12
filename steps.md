# DeepStream Setup Steps

## System Info
- Ubuntu 22.04
- RTX 4060 (Compute 8.9)
- CUDA 12.8
- Driver 591.59
- DeepStream 7.1

---

## Step 1: Install DeepStream SDK

Reference: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html

Download from: https://developer.nvidia.com/deepstream-getting-started

```bash
sudo apt-get install ./deepstream-7.1_7.1.0-1_amd64.deb
```

Verify:
```bash
deepstream-app --version
# Should show: deepstream-app version 7.1.0
```

---

## Step 2: Install Dependencies

```bash
sudo apt install python3-gi python3-gst-1.0 python-gi-dev \
    gir1.2-gst-plugins-base-1.0 gir1.2-gstreamer-1.0 -y
```

---

## Step 3: Recreate venv with System Site Packages

Required so Python can access system-installed `gi` module:

```bash
cd ~/work/tenssort_inference
python3 -m venv venv --system-site-packages --clear
source venv/bin/activate
```

Verify GStreamer access:
```bash
python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('GStreamer OK')"
```

---

## Step 4: Build and Install pyds (Python Bindings)

```bash
# Clone the repo
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git /tmp/deepstream_python_apps

# Checkout version for DeepStream 7.1
cd /tmp/deepstream_python_apps
git checkout v1.2.0

# Initialize submodules
cd bindings
git submodule update --init

# Build
mkdir -p build && cd build
cmake .. -DDS_VERSION=7.1
make -j$(nproc)

# Install
pip install ../
```

Verify:
```bash
python3 -c "import pyds; print('pyds installed')"
```

---

## Step 5: Reinstall pip packages

After recreating venv, reinstall your project dependencies:

```bash
pip install torch torchvision
pip install opencv-python numpy pyyaml
# ... other packages as needed
```

---

## Next Steps

- [ ] Create directory structure in `deepstream/`
- [ ] Create DeepStream config files
- [ ] Create custom YOLO parser (if needed)
- [ ] Create Python pipeline script

