import numpy as np

def feature_extraction(frame):
    gray = rgb_to_grayscale(frame)
    features = detect_edges(gray)
    return features

def rgb_to_grayscale(frame):
    return 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]

def detect_edges(gray):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    grad_x = convolve(gray, sobel_x)
    grad_y = convolve(gray, sobel_y)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    edges = grad > 100
    return edges

def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
    convolved = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            convolved[i,j] = np.sum(region * kernel)
    return convolved

def feature_matching(features):
    matches = []
    for i in range(len(features)-1):
        match = match_features(features[i], features[i+1])
        matches.append(match)
    return matches

def match_features(feat1, feat2):
    points1 = np.argwhere(feat1)
    points2 = np.argwhere(feat2)
    matches = []
    for p1 in points1:
        distances = np.linalg.norm(points2 - p1, axis=1)
        if len(distances) == 0:
            continue
        min_idx = np.argmin(distances)
        if distances[min_idx] < 50:
            matches.append([p1, points2[min_idx]])
    return matches
