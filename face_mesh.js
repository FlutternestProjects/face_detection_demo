// MediaPipe Face Mesh Implementation
// This implementation provides 468 facial landmarks with high accuracy and performance

// Store initialized state
let faceMeshModel = null;
let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let results = null;
let camera = null;
let isProcessing = false;
let debug = false;

// Face mesh thresholds for detection quality
const MIN_DETECTION_CONFIDENCE = 0.5;
const MIN_TRACKING_CONFIDENCE = 0.5;

// State variables for position detection
const LEFT_THRESHOLD = 0.42;
const RIGHT_THRESHOLD = 0.58;
const FACE_MIN_SIZE_RATIO = 0.15; // Minimum face size as ratio of image width
const FACE_MAX_SIZE_RATIO = 0.3; // Maximum face size as ratio of image width
const TILT_THRESHOLD = 15; // In degrees

// Head rotation thresholds
const HEAD_ROTATION_THRESHOLD = 0.5; // Cosine threshold for ~45 degrees

/**
 * Initialize face mesh detection
 * @param {HTMLVideoElement} video - Video element to use for detection
 * @param {Object} options - Configuration options
 * @param {Function} callback - Callback function to call when initialization is complete
 */
function initFaceMesh(video, options = {}, callback) {
  console.log("Initializing MediaPipe Face Mesh...");
  
  if (!window.FaceMesh) {
    console.error("MediaPipe FaceMesh is not loaded. Make sure to include the MediaPipe library.");
    callback(false, "MediaPipe FaceMesh library not found");
    return;
  }
  
  if (!window.Camera) {
    console.error("MediaPipe Camera is not loaded. Make sure to include the MediaPipe Camera Utils.");
    callback(false, "MediaPipe Camera library not found");
    return;
  }
  
  videoElement = video;
  debug = options.debug || false;
  
  // Mirror the video horizontally
  videoElement.style.transform = 'scaleX(-1)';
  
  // Setup canvas if debug is enabled
  if (debug) {
    canvasElement = document.createElement('canvas');
    canvasElement.width = videoElement.width || 640;
    canvasElement.height = videoElement.height || 480;
    canvasElement.style.position = 'absolute';
    canvasElement.style.top = '0';
    canvasElement.style.left = '0';
    canvasElement.style.zIndex = '10';
    canvasElement.style.transform = 'scaleX(-1)'; // Mirror the debug canvas as well
    document.body.appendChild(canvasElement);
    canvasCtx = canvasElement.getContext('2d');
  }
  
  // Initialize the FaceMesh model
  try {
    faceMeshModel = new window.FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });
    
    faceMeshModel.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: MIN_DETECTION_CONFIDENCE,
      minTrackingConfidence: MIN_TRACKING_CONFIDENCE
    });
    
    faceMeshModel.onResults(onResults);
    
    // Initialize camera using MediaPipe Camera utility
    // This prevents the video from freezing during processing
    camera = new window.Camera(videoElement, {
      onFrame: async () => {
        if (!isProcessing) {
          isProcessing = true;
          await faceMeshModel.send({image: videoElement});
        }
      },
      width: 1280,
      height: 720
    });
    
    camera.start()
      .then(() => {
        console.log("MediaPipe Face Mesh initialized successfully");
        callback(true);
      })
      .catch((error) => {
        console.error("Error starting camera:", error);
        callback(false, error.message);
      });
  } catch (error) {
    console.error("Error initializing MediaPipe Face Mesh:", error);
    callback(false, error.message);
  }
}

/**
 * Process MediaPipe face mesh results
 * @param {Object} detectionResults - Results from MediaPipe face mesh
 */
function onResults(detectionResults) {
  results = detectionResults;
  isProcessing = false;
  
  /* Commenting out debug visualization
  if (debug && canvasCtx) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw video frame
    canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
    
    // Draw face mesh
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      for (const landmarks of results.multiFaceLandmarks) {
        window.drawConnectors(canvasCtx, landmarks, window.FACEMESH_TESSELATION,
                       {color: '#C0C0C070', lineWidth: 1});
        window.drawConnectors(canvasCtx, landmarks, window.FACEMESH_RIGHT_EYE, 
                       {color: '#FF3030'});
        window.drawConnectors(canvasCtx, landmarks, window.FACEMESH_LEFT_EYE,
                       {color: '#30FF30image.png'});
        window.drawConnectors(canvasCtx, landmarks, window.FACEMESH_FACE_OVAL,
                       {color: '#E0E0E0'});
        window.drawConnectors(canvasCtx, landmarks, window.FACEMESH_LIPS,
                       {color: '#E0E0E0'});
      }
    }
    
    canvasCtx.restore();
  }
  */
}

/**
 * Detect face in the current video frame
 * @param {Function} callback - Callback function with detection results
 */
function detectFace(callback) {
  try {
    if (!faceMeshModel || !videoElement) {
      callback(null, "Face mesh not initialized properly");
      return;
    }
    
    // Return the last results - camera is continuously processing frames
    if (results && results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      try {
        const faceMeasurements = calculateFacePosition(
          results.multiFaceLandmarks[0],
          videoElement.videoWidth || 1280,
          videoElement.videoHeight || 720
        );
        
        if (faceMeasurements) {
          callback(faceMeasurements);
        } else {
          callback(null, "Error calculating face position");
        }
      } catch (error) {
        console.error("Error in face detection:", error);
        callback(null, "Error processing face landmarks: " + error.message);
      }
    } else {
      callback(null, "No face detected");
    }
  } catch (error) {
    console.error("Unexpected error in detectFace:", error);
    callback(null, "Unexpected error: " + error.message);
  }
}

/**
 * Calculate face position based on landmarks
 * @param {Array} landmarks - Face landmarks
 * @param {number} imageWidth - Width of the image
 * @param {number} imageHeight - Height of the image
 * @returns {Object} Face position information
 */
function calculateFacePosition(landmarks, imageWidth, imageHeight) {
  try {
    // Key landmarks for face orientation
    const noseTip = landmarks[1];     // Nose tip
    const noseBridge = landmarks[168]; // Bridge of nose
    const leftEye = landmarks[33];     // Left eye outer corner
    const rightEye = landmarks[263];   // Right eye outer corner
    const leftCheek = landmarks[234];  // Left cheek
    const rightCheek = landmarks[454]; // Right cheek
    const topForehead = landmarks[10]; // Top of forehead
    const bottomChin = landmarks[152]; // Bottom of chin
    const leftJaw = landmarks[234];    // Left jaw
    const rightJaw = landmarks[454];   // Right jaw
    
    if (!noseTip || !noseBridge || !leftEye || !rightEye || !topForehead || !bottomChin) {
      console.error("Missing key facial landmarks");
      return null;
    }

    // Calculate face normal vector using multiple points
    const v1 = {
      x: rightEye.x - leftEye.x,
      y: rightEye.y - leftEye.y,
      z: rightEye.z - leftEye.z
    };
    
    const v2 = {
      x: noseTip.x - noseBridge.x,
      y: noseTip.y - noseBridge.y,
      z: noseTip.z - noseBridge.z
    };

    // Calculate face normal vector (cross product)
    const normal = {
      x: v1.y * v2.z - v1.z * v2.y,
      y: v1.z * v2.x - v1.x * v2.z,
      z: v1.x * v2.y - v1.y * v2.x
    };

    // Normalize the vector
    const length = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;

    const forwardProjection = normal.z;
    const sideProjection = normal.x;

    // Define separate thresholds for left and right
    const LEFT_THRESHOLD = 0.7;  // Requires strong left rotation
    const RIGHT_THRESHOLD = -0.7; // Requires strong right rotation

    // Determine face position with separate thresholds
    let position;
    if (sideProjection > LEFT_THRESHOLD) {
      position = "right"; // Swapped from left to right due to mirroring
    } else if (sideProjection < RIGHT_THRESHOLD) {
      position = "left";  // Swapped from right to left due to mirroring
    } else {
      position = "center";
    }

    // Calculate multiple face measurements for robust distance detection
    const faceWidth = Math.abs(rightCheek.x - leftCheek.x);
    const faceHeight = Math.abs(topForehead.y - bottomChin.y);
    const eyeDistance = Math.sqrt(
      Math.pow(rightEye.x - leftEye.x, 2) +
      Math.pow(rightEye.y - leftEye.y, 2) +
      Math.pow(rightEye.z - leftEye.z, 2)
    );
    const jawWidth = Math.sqrt(
      Math.pow(rightJaw.x - leftJaw.x, 2) +
      Math.pow(rightJaw.y - leftJaw.y, 2) +
      Math.pow(rightJaw.z - leftJaw.z, 2)
    );

    // Use a weighted average of multiple measurements
    // This makes the distance detection more robust during head rotation
    const weightedSize = (
      (faceHeight * 0.4) +  // Height is less affected by rotation
      (eyeDistance * 0.3) + // Eye distance provides good stability
      (jawWidth * 0.3)      // Jaw width helps with side angles
    );

    // console.log("Distance metrics:", {
    //   faceWidth,
    //   faceHeight,
    //   eyeDistance,
    //   jawWidth,
    //   weightedSize
    // });

    // Adjusted thresholds for the weighted measurement
    const MIN_WEIGHTED_SIZE = 0.25;
    const MAX_WEIGHTED_SIZE = 0.45;
    
    // Enhanced distance status using weighted size
    let distanceStatus;
    if (weightedSize < MIN_WEIGHTED_SIZE) {
      distanceStatus = "too_far";
    } else if (weightedSize > MAX_WEIGHTED_SIZE) {
      distanceStatus = "too_close";
    } else {
      distanceStatus = "good";
    }

    // Calculate face tilt angle
    const deltaY = rightEye.y - leftEye.y;
    const deltaX = rightEye.x - leftEye.x;
    const tiltAngle = Math.atan2(deltaY, deltaX) * (180 / Math.PI);
    const isLevel = Math.abs(tiltAngle) < TILT_THRESHOLD;
    
    // Calculate bounding box
    const minX = Math.min(...landmarks.map(l => l.x)) * imageWidth;
    const maxX = Math.max(...landmarks.map(l => l.x)) * imageWidth;
    const minY = Math.min(...landmarks.map(l => l.y)) * imageHeight;
    const maxY = Math.max(...landmarks.map(l => l.y)) * imageHeight;
    
    // Adjust bounding box for mirrored display
    const boundingBox = {
      left: imageWidth - maxX, // Flip horizontal position
      top: minY,
      width: maxX - minX,
      height: maxY - minY
    };
    
    // Log detailed information for debugging
    if (debug) {
      console.log(`Face Detection Details:
      - forwardProjection: ${forwardProjection.toFixed(4)}
      - sideProjection: ${sideProjection.toFixed(4)}
      - position: ${position}
      - weightedSize: ${weightedSize.toFixed(3)}
      - distanceStatus: ${distanceStatus}
      - tiltAngle: ${tiltAngle.toFixed(2)}°
      `);
    }
    
    return {
      normalizedX: noseTip.x,
      position,
      faceWidth,
      faceHeight,
      tiltAngle,
      isLevel,
      isGoodDistance: distanceStatus === "good",
      distanceStatus,
      boundingBox,
      sideProjection
    };
  } catch (error) {
    console.error("Error in calculateFacePosition:", error);
    return null;
  }
}

/**
 * Clean up resources
 */
function disposeFaceMesh() {
  if (camera) {
    camera.stop();
    camera = null;
  }
  
  if (faceMeshModel) {
    faceMeshModel.close();
    faceMeshModel = null;
  }
  
  if (debug && canvasElement) {
    document.body.removeChild(canvasElement); 
    canvasElement = null;
    canvasCtx = null;
  }
  
  videoElement = null;
  results = null;
  isProcessing = false;
}

// Export functions for Dart interop
window.faceMeshDetection = {
  initFaceMesh: initFaceMesh,
  detectFace: detectFace,
  disposeFaceMesh: disposeFaceMesh
}; 