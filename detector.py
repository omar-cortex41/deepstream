#!/usr/bin/env python3
"""
DeepStream Multi-Stream Pipeline with FPS Metrics

Configuration files (in config/):
  - config_infer_primary.txt  : Inference settings (model, batch size, etc.)
  - sources.txt               : Video sources (one per line)
  - labels.txt                : Class labels
"""
import sys
import os
import time
from pathlib import Path
from collections import deque

SCRIPT_DIR = Path(__file__).parent.absolute()
CONFIG_DIR = SCRIPT_DIR.parent / "config"
os.chdir(SCRIPT_DIR)

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds


# ============================================================================
# Pipeline Settings
# ============================================================================
INFER_CONFIG = str(CONFIG_DIR / "config_infer_primary.txt")
SOURCES_FILE = str(CONFIG_DIR / "sources.txt")

# Streammux should match source aspect ratio (16:9), not model input (640x640)
# nvinfer handles resize to model input internally
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
STREAMMUX_TIMEOUT = 40000

# Tiler settings - computed dynamically based on num_sources
# Each cell maintains 16:9 aspect ratio
TILER_OUTPUT_WIDTH = 1920  # Total output width

DISPLAY = True    # Set False for headless
SYNC = False      # Set True to sync to video framerate

# Output settings
# OUTPUT_FILE = None  # Set to path like "output.mp4" to save, None to disable
OUTPUT_FILE = "../../output/deepstream_output.mp4"


class Metrics:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_window = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.last_fps_count = 0
        self.detection_count = 0
        
    def update(self, num_frames: int, num_detections: int):
        self.frame_count += num_frames
        self.detection_count += num_detections
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            fps = (self.frame_count - self.last_fps_count) / elapsed
            self.fps_window.append(fps)
            self.last_fps_count = self.frame_count
            self.last_fps_time = now
            
    @property
    def current_fps(self) -> float:
        return self.fps_window[-1] if self.fps_window else 0.0
    
    @property
    def avg_fps(self) -> float:
        return sum(self.fps_window) / len(self.fps_window) if self.fps_window else 0.0
    
    @property
    def total_time(self) -> float:
        return time.time() - self.start_time


metrics = Metrics()
num_sources = 0


def compute_tiler_grid(n: int) -> tuple:
    """Compute optimal tiler grid (rows, cols, width, height) for n sources.

    Maintains 16:9 aspect ratio per cell.
    """
    import math

    # Compute grid dimensions
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Each cell should be 16:9
    # Total width = TILER_OUTPUT_WIDTH, so cell_width = width / cols
    cell_width = TILER_OUTPUT_WIDTH // cols
    cell_height = cell_width * 9 // 16  # Maintain 16:9

    total_width = cell_width * cols
    total_height = cell_height * rows

    return rows, cols, total_width, total_height


def load_sources(filepath: str) -> list:
    """Load sources from config file."""
    sources = []
    config_dir = Path(filepath).parent
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Convert to URI
            if line.startswith('rtsp://') or line.startswith('file://'):
                uri = line
            else:
                # Relative path - resolve from config dir
                path = (config_dir / line).resolve()
                if not path.exists():
                    print(f"Warning: File not found: {path}")
                    continue
                uri = f"file://{path}"
            
            sources.append(uri)
    
    return sources


def pgie_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    frame_count = 0
    detection_count = 0

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_count += 1
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    detection_count += 1
                    l_obj = l_obj.next
                except StopIteration:
                    break
            l_frame = l_frame.next
        except StopIteration:
            break

    metrics.update(frame_count, detection_count)
    
    if metrics.frame_count % 100 == 0:
        per_stream = metrics.current_fps / num_sources if num_sources else 0
        print(f"\rFPS: {metrics.current_fps:.1f} total | {per_stream:.1f}/stream | "
              f"Avg: {metrics.avg_fps:.1f} | Frames: {metrics.frame_count} | "
              f"Dets: {metrics.detection_count}", end='', flush=True)

    return Gst.PadProbeReturn.OK


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print(f"\n\nEnd of stream")
        print(f"Total frames: {metrics.frame_count}")
        print(f"Total time: {metrics.total_time:.1f}s")
        print(f"Average FPS: {metrics.avg_fps:.1f}")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\nError: {err}, {debug}")
        loop.quit()
    return True


def cb_newpad(decodebin, decoder_src_pad, data):
    streammux, stream_id = data
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps(None)
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    
    if gstname.find("video") != -1:
        sinkpad = streammux.request_pad_simple(f"sink_{stream_id}")
        if sinkpad and not sinkpad.is_linked():
            decoder_src_pad.link(sinkpad)


def decodebin_child_added(child_proxy, obj, name, user_data):
    if "decodebin" in name:
        obj.connect("child-added", decodebin_child_added, user_data)
    if "nvv4l2decoder" in name:
        obj.set_property("gpu-id", 0)


def create_source_bin(index: int, uri: str, streammux):
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        return None
    
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, (streammux, index))
    uri_decode_bin.connect("child-added", decodebin_child_added, None)
    return uri_decode_bin


def main():
    global num_sources
    
    # Load sources
    sources = load_sources(SOURCES_FILE)
    if not sources:
        print(f"Error: No sources found in {SOURCES_FILE}")
        sys.exit(1)
    
    num_sources = len(sources)
    is_live = any('rtsp://' in s for s in sources)
    
    print("=" * 60)
    print("DeepStream Multi-Stream Pipeline")
    print("=" * 60)
    print(f"Sources: {num_sources}")
    print(f"Live: {is_live}")
    print("=" * 60)
    
    Gst.init(None)
    pipeline = Gst.Pipeline()
    
    # Stream muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("width", STREAMMUX_WIDTH)
    streammux.set_property("height", STREAMMUX_HEIGHT)
    streammux.set_property("batched-push-timeout", STREAMMUX_TIMEOUT)
    streammux.set_property("gpu-id", 0)
    streammux.set_property("live-source", is_live)
    pipeline.add(streammux)
    
    # Add sources
    for i, uri in enumerate(sources):
        print(f"  [{i}] {uri}")
        source_bin = create_source_bin(i, uri, streammux)
        if source_bin:
            pipeline.add(source_bin)
    
    # Inference
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", INFER_CONFIG)
    pgie.set_property("batch-size", num_sources)
    pipeline.add(pgie)
    
    pgiesrcpad = pgie.get_static_pad("src")
    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)
    
    # Tiler + display (dynamic grid based on num_sources)
    tiler_rows, tiler_cols, tiler_width, tiler_height = compute_tiler_grid(num_sources)
    print(f"Tiler: {tiler_cols}x{tiler_rows} grid, {tiler_width}x{tiler_height} output")

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_cols)
    tiler.set_property("width", tiler_width)
    tiler.set_property("height", tiler_height)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "converter")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Build sink chain
    elements = [tiler, nvvidconv, nvosd]

    if OUTPUT_FILE:
        # File output: nvosd -> nvvidconv2 -> encoder -> parser -> mux -> filesink
        output_path = Path(OUTPUT_FILE)
        if not output_path.is_absolute():
            output_path = (SCRIPT_DIR / OUTPUT_FILE).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Output: {output_path}")

        nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        encoder.set_property("bitrate", 8000000)
        parser = Gst.ElementFactory.make("h264parse", "parser")
        mux = Gst.ElementFactory.make("mp4mux", "mux")
        filesink = Gst.ElementFactory.make("filesink", "filesink")
        filesink.set_property("location", str(output_path))
        filesink.set_property("sync", False)

        elements.extend([nvvidconv2, encoder, parser, mux, filesink])

        if DISPLAY:
            # Tee to both file and display
            tee = Gst.ElementFactory.make("tee", "tee")
            queue1 = Gst.ElementFactory.make("queue", "queue1")
            queue2 = Gst.ElementFactory.make("queue", "queue2")
            sink = Gst.ElementFactory.make("nveglglessink", "sink")
            sink.set_property("sync", SYNC)

            for elem in [tiler, nvvidconv, nvosd, tee, queue1, queue2,
                         nvvidconv2, encoder, parser, mux, filesink, sink]:
                pipeline.add(elem)

            streammux.link(pgie)
            pgie.link(tiler)
            tiler.link(nvvidconv)
            nvvidconv.link(nvosd)
            nvosd.link(tee)

            # Branch 1: display
            tee.link(queue1)
            queue1.link(sink)

            # Branch 2: file
            tee.link(queue2)
            queue2.link(nvvidconv2)
            nvvidconv2.link(encoder)
            encoder.link(parser)
            parser.link(mux)
            mux.link(filesink)
        else:
            # File only
            for elem in elements:
                pipeline.add(elem)

            streammux.link(pgie)
            pgie.link(tiler)
            tiler.link(nvvidconv)
            nvvidconv.link(nvosd)
            nvosd.link(nvvidconv2)
            nvvidconv2.link(encoder)
            encoder.link(parser)
            parser.link(mux)
            mux.link(filesink)
    else:
        # Display or fakesink only
        if DISPLAY:
            sink = Gst.ElementFactory.make("nveglglessink", "sink")
        else:
            sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", SYNC)
        elements.append(sink)

        for elem in elements:
            pipeline.add(elem)

        streammux.link(pgie)
        pgie.link(tiler)
        tiler.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)
    
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    print("\nStarting pipeline... Press Ctrl+C to stop")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print(f"\n\nInterrupted")
        print(f"Total frames: {metrics.frame_count}")
        print(f"Total time: {metrics.total_time:.1f}s")
        print(f"Average FPS: {metrics.avg_fps:.1f}")
    
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")


if __name__ == '__main__':
    main()
