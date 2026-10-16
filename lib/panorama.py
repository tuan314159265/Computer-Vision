"""Pipeline ghép ảnh toàn cảnh dựa trên đặc trưng (SIFT / ORB).

Các bước: tiền xử lý -> detect & describe -> match (Lowe ratio)
-> homography (RANSAC) -> warp về ảnh giữa -> blending -> crop viền đen.

Nội dung tương đương phần 2 của notebook.
"""
import time

import cv2
import numpy as np

__all__ = [
    'preprocess_for_features',
    'detect_and_describe',
    'match_features',
    'estimate_homography',
    'compute_cumulative_homographies',
    'crop_black_borders',
    'warp_and_blend',
    'compute_overlap_photometric_error',
    'stitch',
]


def preprocess_for_features(rgb_image):
    """
    Tiền xử lý ảnh phục vụ phát hiện đặc trưng:
    - Kiểm tra số kênh và chuyển đổi sang ảnh xám 8-bit (CV_8U).
    - Đảm bảo ma trận dữ liệu liền mạch trong bộ nhớ (contiguous array).
    """
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng hoặc không hợp lệ.")

    if rgb_image.ndim == 3:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_image.copy()

    return np.ascontiguousarray(gray, dtype=np.uint8)


def detect_and_describe(gray_image, method='SIFT', max_features=3000):
    """
    Trích xuất keypoints và descriptors cho ảnh xám.

    Tham số:
        gray_image: Ảnh xám uint8.
        method: 'SIFT' hoặc 'ORB'.
        max_features: Giới hạn số điểm đặc trưng giữ lại (với ORB).
    """
    method = method.upper()

    if method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=max_features)
    elif method == 'ORB':
        detector = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            fastThreshold=20
        )
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {method}. Vui lòng chọn 'SIFT' hoặc 'ORB'.")

    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    if descriptors is None or len(keypoints) == 0:
        desc_dim = 128 if method == 'SIFT' else 32
        desc_dtype = np.float32 if method == 'SIFT' else np.uint8
        descriptors = np.empty((0, desc_dim), dtype=desc_dtype)
        keypoints = []

    return keypoints, descriptors


def match_features(desc1, desc2, method='SIFT', ratio_thresh=0.75):
    """
    So khớp đặc trưng k-NN (k=2) bằng Brute-Force Matcher:
    - SIFT: khoảng cách L2 (cv2.NORM_L2)
    - ORB : khoảng cách Hamming (cv2.NORM_HAMMING)
    Lọc match bằng tỉ số khoảng cách Lowe: d1 < ratio_thresh * d2
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    method = method.upper()
    if method == "SIFT":
        norm_type = cv2.NORM_L2
    elif method == "ORB":
        norm_type = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {method}. Vui lòng chọn 'SIFT' hoặc 'ORB'.")

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    return good_matches


def estimate_homography(kps1, kps2, matches, ransac_thresh=3.0):
    """
    Ước lượng ma trận Homography H ánh xạ từ ảnh 2 sang ảnh 1
    (x1 ~ H * x2) bằng RANSAC.

    Trả về:
    - H: ma trận Homography.
    - inlier_mask: mask xác định các match là inlier.
    - reproj_error: sai số chiếu lại trung bình (MAE) trên các inlier.
    """
    if len(matches) < 4:
        return None, np.array([], dtype=bool), float('nan')

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None, np.array([], dtype=bool), float('nan')

    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() == 0:
        return H, inlier_mask, float('nan')

    proj_pts2 = cv2.perspectiveTransform(pts2[inlier_mask], H)
    errors = np.linalg.norm(pts1[inlier_mask] - proj_pts2, axis=2).ravel()
    reproj_error = float(np.mean(errors))

    return H, inlier_mask, reproj_error


def compute_cumulative_homographies(pair_homographies, n_images):
    """
    Tích lũy ma trận biến đổi để đưa mọi ảnh về hệ tọa độ của ảnh giữa c = n // 2:
    - Ảnh giữa: T_c = I
    - Các ảnh bên phải (i > c): T_i = T_{i-1} * H_{i -> i-1}
    - Các ảnh bên trái  (i < c): T_i = T_{i+1} * inv(H_{i+1 -> i})
    """
    c = n_images // 2
    T = [np.eye(3, dtype=np.float64) for _ in range(n_images)]

    for i in range(c + 1, n_images):
        H_curr = pair_homographies[i - 1]
        if H_curr is None or T[i - 1] is None:
            T[i] = None
        else:
            T[i] = T[i - 1] @ H_curr

    for i in range(c - 1, -1, -1):
        H_next = pair_homographies[i]
        if H_next is None or T[i + 1] is None:
            T[i] = None
        else:
            T[i] = T[i + 1] @ np.linalg.inv(H_next)

    return T, c


def crop_black_borders(img_rgb):
    """Cắt bounding box bao quanh các pixel hợp lệ để loại bỏ viền đen."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 0
    if not np.any(mask):
        return img_rgb
    ys, xs = np.where(mask)
    return img_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def warp_and_blend(images, cumulative_homographies, blend_mode='feather'):
    """
    Warp các ảnh lên canvas chung và thực hiện hòa trộn:
    - 'overwrite': ảnh sau đè ảnh trước
    - 'average'  : trung bình cộng tại các pixel chồng lấn
    - 'feather'  : trọng số tuyến tính theo Distance Transform (mượt biên)
    """
    all_corners = []
    for img, H in zip(images, cumulative_homographies):
        if H is None:
            continue
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_c = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_c)

    if not all_corners:
        return None, []

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0)[0]).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)[0]).astype(int)

    canvas_w = int(x_max - x_min)
    canvas_h = int(y_max - y_min)

    T_shift = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0,      1]], dtype=np.float64)

    final_Hs = [T_shift @ H if H is not None else None for H in cumulative_homographies]

    warped_imgs = []
    masks = []
    weights = []

    for img, H in zip(images, final_Hs):
        if H is None:
            continue
        warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
        mask = (cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)

        warped_imgs.append(warped)
        masks.append(mask)

        if blend_mode == 'feather':
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            weights.append(dist)

    if blend_mode == 'overwrite':
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for warped, mask in zip(warped_imgs, masks):
            canvas[mask > 0] = warped[mask > 0]

    elif blend_mode == 'average':
        acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        for warped, mask in zip(warped_imgs, masks):
            m_f = mask[..., None].astype(np.float32)
            acc += warped.astype(np.float32) * m_f
            count += m_f
        canvas = np.where(count > 0, acc / np.maximum(count, 1.0), 0)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    elif blend_mode == 'feather':
        acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        total_w = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        for warped, w_map in zip(warped_imgs, weights):
            w_3d = w_map[..., None].astype(np.float32)
            acc += warped.astype(np.float32) * w_3d
            total_w += w_3d
        canvas = np.where(total_w > 0, acc / np.maximum(total_w, 1e-6), 0)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Chế độ blending không hỗ trợ: {blend_mode}")

    cropped_canvas = crop_black_borders(canvas)
    return cropped_canvas, warped_imgs


def compute_overlap_photometric_error(img1_warped, img2_warped):
    """
    Tính sai số quang học vùng chồng lấn (Overlap Photometric Error):
    MAE giữa hai ảnh xám sau khi chuẩn hóa trừ đi độ sáng trung bình.
    """
    g1 = cv2.cvtColor(img1_warped, cv2.COLOR_RGB2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2_warped, cv2.COLOR_RGB2GRAY).astype(np.float64)
    overlap = (g1 > 0) & (g2 > 0)

    if overlap.sum() < 200:
        return float('nan')

    v1 = g1[overlap] - g1[overlap].mean()
    v2 = g2[overlap] - g2[overlap].mean()
    return float(np.mean(np.abs(v1 - v2)))


def _opencv_stitch_fallback(images, mode=cv2.Stitcher_PANORAMA):
    """Use OpenCV's panorama or flat-scan stitcher for an unreliable chain."""
    stitcher = cv2.Stitcher_create(mode)
    bgr_images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
    status, panorama_bgr = stitcher.stitch(bgr_images)
    if status != cv2.Stitcher_OK or panorama_bgr is None:
        return None, status
    return cv2.cvtColor(panorama_bgr, cv2.COLOR_BGR2RGB), status


def _best_reliable_image_order(features, method, min_inliers=8, min_inlier_ratio=0.15):
    """Find the strongest reliable Hamiltonian path from all pairwise matches.

    This is attempted only after the supplied order contains a weak adjacent
    pair. Dynamic programming keeps the search practical for small image sets.
    """
    n_images = len(features)
    if n_images < 3 or n_images > 12:
        return None

    edge_scores = np.full((n_images, n_images), -np.inf, dtype=np.float64)
    for left in range(n_images):
        _, desc_left = features[left]
        for right in range(left + 1, n_images):
            keypoints_left, desc_left = features[left]
            keypoints_right, desc_right = features[right]
            matches = match_features(desc_left, desc_right, method=method)
            _, inlier_mask, reproj_error = estimate_homography(
                keypoints_left, keypoints_right, matches
            )
            n_matches = len(matches)
            n_inliers = int(inlier_mask.sum()) if len(inlier_mask) else 0
            inlier_ratio = n_inliers / n_matches if n_matches else 0.0
            if (n_inliers >= min_inliers and inlier_ratio >= min_inlier_ratio
                    and np.isfinite(reproj_error)):
                score = np.log1p(n_inliers) + inlier_ratio
                edge_scores[left, right] = score
                edge_scores[right, left] = score

    # (visited image bitmask, final image) -> (score, order tuple)
    paths = {
        (1 << image, image): (0.0, (image,))
        for image in range(n_images)
    }
    for visited in range(1 << n_images):
        for last in range(n_images):
            current = paths.get((visited, last))
            if current is None:
                continue
            score, order = current
            for next_image in range(n_images):
                bit = 1 << next_image
                edge_score = edge_scores[last, next_image]
                if visited & bit or not np.isfinite(edge_score):
                    continue
                key = (visited | bit, next_image)
                candidate = (score + edge_score, order + (next_image,))
                if key not in paths or candidate[0] > paths[key][0]:
                    paths[key] = candidate

    full_mask = (1 << n_images) - 1
    complete_paths = [paths[(full_mask, last)] for last in range(n_images)
                      if (full_mask, last) in paths]
    return list(max(complete_paths, key=lambda item: item[0])[1]) if complete_paths else None


def _estimate_adjacent_pairs(features, method, min_inliers=8, min_inlier_ratio=0.15):
    """Estimate adjacent homographies and identify unreliable image pairs."""
    homographies = []
    pair_metrics = []
    weak_pairs = []

    for index in range(len(features) - 1):
        kps1, desc1 = features[index]
        kps2, desc2 = features[index + 1]
        matches = match_features(desc1, desc2, method=method)
        homography, inlier_mask, reproj_error = estimate_homography(kps1, kps2, matches)

        n_matches = len(matches)
        n_inliers = int(inlier_mask.sum()) if len(inlier_mask) else 0
        inlier_ratio = n_inliers / n_matches if n_matches else 0.0
        if (homography is None or n_inliers < min_inliers
                or inlier_ratio < min_inlier_ratio or not np.isfinite(reproj_error)):
            weak_pairs.append(f'{index + 1}-{index + 2}')

        homographies.append(homography)
        pair_metrics.append({
            'pair': f'{index + 1}-{index + 2}',
            'kps_left': len(kps1),
            'kps_right': len(kps2),
            'matches': n_matches,
            'inliers': n_inliers,
            'inlier_ratio': round(inlier_ratio, 3),
            'reproj_error': round(reproj_error, 3),
        })

    return homographies, pair_metrics, weak_pairs


def stitch(images, method='SIFT', blend_mode='feather'):
    """
    Nhận chuỗi ảnh (trái -> phải), chạy pipeline hoàn chỉnh:
    Tiền xử lý -> Detect & Describe -> Match -> RANSAC Homography -> Warp -> Blend.
    """
    start_time = time.time()
    n = len(images)
    grays = [preprocess_for_features(im) for im in images]

    features = [detect_and_describe(gray, method=method) for gray in grays]
    working_images = list(images)
    image_order = list(range(n))
    pair_homographies, pair_metrics, weak_pairs = _estimate_adjacent_pairs(features, method)
    backend = 'feature_pipeline'

    if weak_pairs:
        # File order may not be the capture order. Build a graph of reliable
        # overlaps and find a path that visits each image exactly once.
        reordered_indices = _best_reliable_image_order(features, method)
        if reordered_indices is not None and reordered_indices != image_order:
            image_order = reordered_indices
            working_images = [images[index] for index in image_order]
            features = [features[index] for index in image_order]
            pair_homographies, pair_metrics, weak_pairs = _estimate_adjacent_pairs(features, method)
            if not weak_pairs:
                # Reordered map/document photos are planar scan-like inputs;
                # OpenCV's SCANS mode chooses seams more reliably than feather
                # blending, which visibly doubles map lines in this dataset.
                panorama, opencv_status = _opencv_stitch_fallback(
                    working_images, mode=cv2.Stitcher_SCANS
                )
                if panorama is not None:
                    summary = {
                        'status': 'SUCCESS',
                        'method': method,
                        'blend_mode': 'opencv_scans',
                        'center_index': None,
                        'backend': 'opencv_scans_reordered',
                        'fallback_reason': 'Reordered by reliable pairwise overlap; planar scan blending',
                        'image_order': [index + 1 for index in image_order],
                        'opencv_status': int(opencv_status),
                        'time': round(time.time() - start_time, 3),
                        'pair_metrics': pair_metrics,
                    }
                    return panorama, summary, []
                backend = 'feature_pipeline_reordered'

    if weak_pairs:
        panorama, opencv_status = _opencv_stitch_fallback(working_images)
        summary = {
            'status': 'SUCCESS' if panorama is not None else 'FAIL',
            'method': method,
            'blend_mode': blend_mode,
            'center_index': None,
            'backend': 'opencv_stitcher_fallback' if panorama is not None else 'feature_pipeline_failed',
            'fallback_reason': f"Unreliable homography for pair(s): {', '.join(weak_pairs)}",
            'image_order': [index + 1 for index in image_order],
            'opencv_status': int(opencv_status),
            'time': round(time.time() - start_time, 3),
            'pair_metrics': pair_metrics,
        }
        return panorama, summary, []

    cum_H, center_idx = compute_cumulative_homographies(pair_homographies, n)
    panorama, warped_imgs = warp_and_blend(working_images, cum_H, blend_mode=blend_mode)

    if warped_imgs and len(warped_imgs) == n:
        for i in range(n - 1):
            pe = compute_overlap_photometric_error(warped_imgs[i], warped_imgs[i + 1])
            pair_metrics[i]['photo_error'] = round(pe, 3)

    summary = {
        'status': 'SUCCESS' if panorama is not None else 'FAIL',
        'method': method,
        'blend_mode': blend_mode,
        'center_index': center_idx + 1,
        'backend': backend,
        'image_order': [index + 1 for index in image_order],
        'time': round(time.time() - start_time, 3),
        'pair_metrics': pair_metrics
    }
    return panorama, summary, warped_imgs
