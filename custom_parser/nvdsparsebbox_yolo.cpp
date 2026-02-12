/*
 * Custom YOLO bounding box parser for DeepStream
 * 
 * Parses Ultralytics YOLO output format:
 * - Shape: [batch, 300, 6]
 * - Format: [x1, y1, x2, y2, confidence, class_id]
 * - Already post-NMS from Ultralytics export
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN_CONFIDENCE 0.25f
#define MAX_DETECTIONS 300

/* Parse bounding boxes from YOLO output */
extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layers found" << std::endl;
        return false;
    }

    // Get output layer - should be [300, 6]
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* output = static_cast<const float*>(layer.buffer);

    if (!output) {
        std::cerr << "ERROR: Output buffer is null" << std::endl;
        return false;
    }

    // Network input dimensions
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    // Parse detections
    // Format: [300, 6] where each row is [x1, y1, x2, y2, conf, class_id]
    const int stride = 6;

    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        const float* det = output + i * stride;

        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float confidence = det[4];
        int classId = static_cast<int>(det[5]);

        // Skip low confidence or invalid detections
        if (confidence < MIN_CONFIDENCE) continue;
        if (x2 <= x1 || y2 <= y1) continue;
        if (classId < 0) continue;

        // Check class filter if specified
        if (detectionParams.numClassesConfigured > 0 &&
            classId >= static_cast<int>(detectionParams.numClassesConfigured)) {
            continue;
        }

        // Check per-class threshold if available
        float threshold = MIN_CONFIDENCE;
        if (classId < static_cast<int>(detectionParams.perClassPreclusterThreshold.size())) {
            threshold = detectionParams.perClassPreclusterThreshold[classId];
        }
        if (confidence < threshold) continue;

        // Create detection object
        NvDsInferParseObjectInfo obj;
        obj.classId = classId;
        obj.detectionConfidence = confidence;

        // Coordinates are already in network input space (640x640)
        // DeepStream will scale them to original frame size
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;

        objectList.push_back(obj);
    }

    return true;
}

/* Check if custom parser is supported */
extern "C" bool NvDsInferParseCustomYoloCheck(
    const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams)
{
    return true;
}

/* Required by DeepStream */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo);

