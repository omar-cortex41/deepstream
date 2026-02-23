#!/usr/bin/env python3
"""
DeepStream Multi-Stream Object Detection Pipeline
==================================================

This script processes multiple video streams simultaneously using NVIDIA DeepStream.
It runs YOLO object detection on all streams and displays them in a grid.

Configuration files (in config/):
  - config_infer_primary.txt  : Model and inference settings
  - sources.txt               : List of video files or RTSP streams
  - labels.txt                : Object class names (person, car, etc.)

How it works:
  1. Loads video sources from sources.txt
  2. Decodes all videos on GPU
  3. Batches frames together for efficient processing
  4. Runs YOLO detection on the batch
  5. Displays all streams in a grid with bounding boxes
"""

import sys
import os
import time
import math
from collections import deque

# Import GStreamer and DeepStream
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds


# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

# Config files (simple paths)
INFER_CONFIG = "config/config_infer_primary.txt"
SOURCES_FILE = "config/sources.txt"

# Video processing settings
VIDEO_WIDTH = 1920      # Width for processing (16:9 aspect ratio)
VIDEO_HEIGHT = 1080     # Height for processing
BATCH_TIMEOUT = 40000   # Milliseconds to wait for batch

# Display settings
DISPLAY = True          # Show window with video grid
SYNC = False            # False = max speed, True = match video framerate

# Output file (set to None to disable file saving)
OUTPUT_FILE = None


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class PerformanceTracker:
    """Tracks FPS and detection counts"""

    def __init__(self):
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()

        # For FPS calculation
        self.fps_history = deque(maxlen=30)  # Keep last 30 FPS measurements
        self.last_fps_check = time.time()
        self.last_frame_count = 0

    def update(self, frames_processed: int, detections_found: int):
        """Update counters with new batch results"""
        self.total_frames += frames_processed
        self.total_detections += detections_found

        # Calculate FPS every second
        now = time.time()
        time_since_last_check = now - self.last_fps_check

        if time_since_last_check >= 1.0:
            frames_since_last_check = self.total_frames - self.last_frame_count
            current_fps = frames_since_last_check / time_since_last_check

            self.fps_history.append(current_fps)
            self.last_frame_count = self.total_frames
            self.last_fps_check = now

    def get_current_fps(self):
        """Get most recent FPS measurement"""
        return self.fps_history[-1] if self.fps_history else 0.0

    def get_average_fps(self):
        """Get average FPS over last 30 seconds"""
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0

    def get_elapsed_time(self):
        """Get total time since start"""
        return time.time() - self.start_time


# Global tracker
performance = PerformanceTracker()
total_sources = 0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_video_sources(filepath: str) -> list:
    """
    Load video sources from config file.

    Reads sources.txt and converts each line to a proper URI.
    Supports:
      - Video files (relative or absolute paths)
      - RTSP streams (rtsp://...)
      - File URIs (file://...)

    Returns list of URIs ready for GStreamer.
    """
    sources = []
    config_dir = os.path.dirname(filepath)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Already a URI? Use as-is
            if line.startswith('rtsp://') or line.startswith('file://'):
                uri = line
            else:
                # It's a file path - convert to absolute path and URI
                video_path = os.path.join(line)
                video_path = os.path.abspath(video_path)
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found: {video_path}")
                    continue
                uri = f"file://{video_path}"

            sources.append(uri)

    return sources


def calculate_grid_layout(num_videos: int) -> tuple:
    """
    Calculate optimal grid layout for displaying multiple videos.

    For example:
      - 4 videos → 2x2 grid
      - 8 videos → 3x3 grid
      - 1 video  → 1x1 grid

    Returns: (rows, columns, total_width, total_height)
    """
    # Calculate grid dimensions (try to make it square-ish)
    columns = math.ceil(math.sqrt(num_videos))
    rows = math.ceil(num_videos / columns)

    # Calculate size of each cell (maintain 16:9 aspect ratio)
    cell_width = VIDEO_WIDTH // columns
    cell_height = cell_width * 9 // 16

    # Calculate total output size
    total_width = cell_width * columns
    total_height = cell_height * rows

    return rows, columns, total_width, total_height


# ============================================================================
# GSTREAMER CALLBACKS
# ============================================================================

def on_detection_results(pad, info, user_data):
    """
    Called every time we get detection results from the AI model.

    This function:
      1. Extracts detection metadata from the buffer
      2. Counts frames and detections
      3. Updates performance metrics
      4. Prints progress every 100 frames
    """
    # Get the buffer containing detection results
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK

    # Extract metadata (this is DeepStream-specific)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))

    # Count frames and detections in this batch
    frames_in_batch = 0
    detections_in_batch = 0

    # Loop through each frame in the batch
    frame_list = batch_meta.frame_meta_list
    while frame_list is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            frames_in_batch += 1

            # Loop through each detected object in this frame
            object_list = frame_meta.obj_meta_list
            while object_list is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(object_list.data)
                    detections_in_batch += 1
                    object_list = object_list.next
                except StopIteration:
                    break

            frame_list = frame_list.next
        except StopIteration:
            break

    # Update performance tracker
    performance.update(frames_in_batch, detections_in_batch)

    # Print progress every 100 frames
    if performance.total_frames % 100 == 0:
        fps_per_stream = performance.get_current_fps() / total_sources if total_sources else 0
        print(f"\rFPS: {performance.get_current_fps():.1f} total | "
              f"{fps_per_stream:.1f}/stream | "
              f"Avg: {performance.get_average_fps():.1f} | "
              f"Frames: {performance.total_frames} | "
              f"Detections: {performance.total_detections}",
              end='', flush=True)

    return Gst.PadProbeReturn.OK


def on_pipeline_message(bus, message, loop):
    """
    Handles pipeline events like errors or end-of-stream.
    """
    msg_type = message.type

    if msg_type == Gst.MessageType.EOS:
        # All videos finished
        print(f"\n\nAll videos finished!")
        print(f"Total frames processed: {performance.total_frames}")
        print(f"Total time: {performance.get_elapsed_time():.1f}s")
        print(f"Average FPS: {performance.get_average_fps():.1f}")
        loop.quit()

    elif msg_type == Gst.MessageType.ERROR:
        # Something went wrong
        error, debug_info = message.parse_error()
        print(f"\nError: {error}")
        print(f"Debug: {debug_info}")
        loop.quit()

    return True


def on_decoder_pad_added(decoder, decoder_pad, user_data):
    """
    Called when decoder creates a new output pad.
    Connects the decoder to the stream muxer.
    """
    muxer, stream_id = user_data

    # Check if this is a video pad
    caps = decoder_pad.get_current_caps()
    if not caps:
        caps = decoder_pad.query_caps(None)

    structure = caps.get_structure(0)
    stream_type = structure.get_name()

    if "video" in stream_type:
        # Connect this video stream to the muxer
        muxer_pad = muxer.request_pad_simple(f"sink_{stream_id}")
        if muxer_pad and not muxer_pad.is_linked():
            decoder_pad.link(muxer_pad)


def on_decoder_child_added(child_proxy, element, name, user_data):
    """
    Called when decoder creates child elements.
    Ensures hardware decoder uses GPU 0.
    """
    if "decodebin" in name:
        element.connect("child-added", on_decoder_child_added, user_data)
    if "nvv4l2decoder" in name:
        element.set_property("gpu-id", 0)


def create_video_source(stream_index: int, video_uri: str, muxer):
    """
    Creates a video source element that can decode any video format.

    Args:
        stream_index: Index of this stream (0, 1, 2, ...)
        video_uri: URI of video (file:// or rtsp://)
        muxer: The stream muxer to connect to

    Returns:
        GStreamer element that decodes the video
    """
    # Create decoder that auto-detects video format
    decoder = Gst.ElementFactory.make("uridecodebin", f"decoder-{stream_index}")
    if not decoder:
        return None

    # Configure decoder
    decoder.set_property("uri", video_uri)

    # Connect callbacks
    decoder.connect("pad-added", on_decoder_pad_added, (muxer, stream_index))
    decoder.connect("child-added", on_decoder_child_added, None)

    return decoder


# ============================================================================
# PIPELINE BUILDING FUNCTIONS
# ============================================================================

def create_stream_muxer(num_streams: int, is_live: bool):
    """
    Creates the stream muxer that combines multiple video streams into batches.

    The muxer:
      - Takes multiple video inputs
      - Resizes them all to the same size
      - Groups them into batches for efficient processing
    """
    muxer = Gst.ElementFactory.make("nvstreammux", "stream-muxer")

    muxer.set_property("batch-size", num_streams)
    muxer.set_property("width", VIDEO_WIDTH)
    muxer.set_property("height", VIDEO_HEIGHT)
    muxer.set_property("batched-push-timeout", BATCH_TIMEOUT)
    muxer.set_property("gpu-id", 0)
    muxer.set_property("live-source", is_live)

    return muxer


def create_inference_engine(num_streams: int):
    """
    Creates the AI inference engine that runs YOLO object detection.

    Reads configuration from config_infer_primary.txt which specifies:
      - Which TensorRT model to use
      - Confidence thresholds
      - Custom parser for YOLO output
    """
    inference = Gst.ElementFactory.make("nvinfer", "inference-engine")

    inference.set_property("config-file-path", INFER_CONFIG)
    inference.set_property("batch-size", num_streams)

    return inference


def create_video_grid(num_streams: int):
    """
    Creates the tiler that arranges multiple video streams in a grid.

    For example, 8 streams become a 3x3 grid.
    """
    rows, cols, width, height = calculate_grid_layout(num_streams)

    print(f"Grid layout: {cols}x{rows} ({width}x{height})")

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "video-grid")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", width)
    tiler.set_property("height", height)

    return tiler


def create_display_elements():
    """
    Creates elements for drawing bounding boxes and displaying video.

    Returns:
      - converter: Converts video format for display
      - osd: On-Screen Display that draws bounding boxes
    """
    converter = Gst.ElementFactory.make("nvvideoconvert", "converter")
    osd = Gst.ElementFactory.make("nvdsosd", "on-screen-display")

    return converter, osd


def main():
    global total_sources

    # ========================================
    # Step 1: Load video sources
    # ========================================
    print("Loading video sources...")
    video_sources = load_video_sources(SOURCES_FILE)

    if not video_sources:
        print(f"Error: No video sources found in {SOURCES_FILE}")
        sys.exit(1)

    total_sources = len(video_sources)
    has_live_streams = any('rtsp://' in uri for uri in video_sources)

    print("=" * 70)
    print("DeepStream Multi-Stream Object Detection")
    print("=" * 70)
    print(f"Number of streams: {total_sources}")
    print(f"Live streams (RTSP): {'Yes' if has_live_streams else 'No'}")
    print("=" * 70)

    # ========================================
    # Step 2: Initialize GStreamer
    # ========================================
    Gst.init(None)
    pipeline = Gst.Pipeline()

    # ========================================
    # Step 3: Create video decoders
    # ========================================
    print("\nAdding video sources:")
    muxer = create_stream_muxer(total_sources, has_live_streams)
    pipeline.add(muxer)

    for i, uri in enumerate(video_sources):
        print(f"  [{i}] {uri}")
        decoder = create_video_source(i, uri, muxer)
        if decoder:
            pipeline.add(decoder)

    # ========================================
    # Step 4: Create AI inference engine
    # ========================================
    print("\nSetting up AI inference...")
    inference = create_inference_engine(total_sources)
    pipeline.add(inference)

    # Attach callback to get detection results
    inference_output = inference.get_static_pad("src")
    inference_output.add_probe(Gst.PadProbeType.BUFFER, on_detection_results, 0)

    # ========================================
    # Step 5: Create video grid and display
    # ========================================
    print("Setting up display...")
    tiler = create_video_grid(total_sources)
    converter, osd = create_display_elements()

    pipeline.add(tiler)
    pipeline.add(converter)
    pipeline.add(osd)

    # ========================================
    # Step 6: Setup output (display and/or file)
    # ========================================

    # Link the main processing chain: muxer → inference → grid → converter → osd
    muxer.link(inference)
    inference.link(tiler)
    tiler.link(converter)
    converter.link(osd)

    # Now handle output based on settings
    if OUTPUT_FILE and DISPLAY:
        # Both display AND save to file
        print(f"Output: Display + File ({OUTPUT_FILE})")

        # Create file output chain
        output_path = os.path.abspath(OUTPUT_FILE)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Splitter to send video to both display and file
        splitter = Gst.ElementFactory.make("tee", "splitter")
        queue_display = Gst.ElementFactory.make("queue", "queue-display")
        queue_file = Gst.ElementFactory.make("queue", "queue-file")

        # Display output
        display_sink = Gst.ElementFactory.make("nveglglessink", "display")
        display_sink.set_property("sync", SYNC)

        # File output (H.264 video in MP4 container)
        file_converter = Gst.ElementFactory.make("nvvideoconvert", "file-converter")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        encoder.set_property("bitrate", 8000000)  # 8 Mbps
        h264_parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        mp4_muxer = Gst.ElementFactory.make("mp4mux", "mp4-muxer")
        file_sink = Gst.ElementFactory.make("filesink", "file-output")
        file_sink.set_property("location", output_path)
        file_sink.set_property("sync", False)

        # Add all elements to pipeline
        for elem in [splitter, queue_display, queue_file, display_sink,
                     file_converter, encoder, h264_parser, mp4_muxer, file_sink]:
            pipeline.add(elem)

        # Connect everything
        osd.link(splitter)

        # Display branch
        splitter.link(queue_display)
        queue_display.link(display_sink)

        # File branch
        splitter.link(queue_file)
        queue_file.link(file_converter)
        file_converter.link(encoder)
        encoder.link(h264_parser)
        h264_parser.link(mp4_muxer)
        mp4_muxer.link(file_sink)

    elif OUTPUT_FILE:
        # Only save to file (no display)
        print(f"Output: File only ({OUTPUT_FILE})")

        output_path = os.path.abspath(OUTPUT_FILE)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # File output chain
        file_converter = Gst.ElementFactory.make("nvvideoconvert", "file-converter")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        encoder.set_property("bitrate", 8000000)
        h264_parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        mp4_muxer = Gst.ElementFactory.make("mp4mux", "mp4-muxer")
        file_sink = Gst.ElementFactory.make("filesink", "file-output")
        file_sink.set_property("location", output_path)
        file_sink.set_property("sync", False)

        # Add and link
        for elem in [file_converter, encoder, h264_parser, mp4_muxer, file_sink]:
            pipeline.add(elem)

        osd.link(file_converter)
        file_converter.link(encoder)
        encoder.link(h264_parser)
        h264_parser.link(mp4_muxer)
        mp4_muxer.link(file_sink)

    elif DISPLAY:
        # Only display (no file)
        print("Output: Display only")

        display_sink = Gst.ElementFactory.make("nveglglessink", "display")
        display_sink.set_property("sync", SYNC)
        pipeline.add(display_sink)

        osd.link(display_sink)

    else:
        # No output (just process, useful for benchmarking)
        print("Output: None (processing only)")

        fake_sink = Gst.ElementFactory.make("fakesink", "fake-output")
        fake_sink.set_property("sync", SYNC)
        pipeline.add(fake_sink)

        osd.link(fake_sink)

    # ========================================
    # Step 7: Start the pipeline
    # ========================================

    # Setup message handler for errors and end-of-stream
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_pipeline_message, loop)

    print("\n" + "=" * 70)
    print("Starting pipeline... Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print(f"\n\nStopped by user")
        print(f"Total frames: {performance.total_frames}")
        print(f"Total time: {performance.get_elapsed_time():.1f}s")
        print(f"Average FPS: {performance.get_average_fps():.1f}")

    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped")


if __name__ == '__main__':
    main()
