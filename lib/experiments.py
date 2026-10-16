"""Experiment orchestration and output generation for both assignment parts."""
import cv2
import numpy as np
import pandas as pd

from lib import utils
from lib.gradient_domain import (
    build_mask,
    crop_to_mask,
    direct_paste,
    object_color_shift,
    poisson_blend,
    save_rgb,
    seam_discontinuity,
    seam_edge_energy,
)
from lib.panorama import stitch


def run_part1(background, source, mask_path, output_dir):
    """Run baseline, custom Poisson variants, and OpenCV seamlessClone."""
    output_dir = output_dir
    mask, mask_method = build_mask(source, mode='auto', mask_path=mask_path)

    src_crop, mask_crop = crop_to_mask(source, mask)
    max_h = int(0.45 * background.shape[0])
    max_w = int(0.45 * background.shape[1])
    scale = min(1.0, max_h / src_crop.shape[0], max_w / src_crop.shape[1])
    if scale < 1.0:
        src_crop = cv2.resize(src_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(
            mask_crop,
            (src_crop.shape[1], src_crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    mask_crop = (mask_crop > 127).astype(np.uint8)

    background_h, background_w = background.shape[:2]
    crop_h, crop_w = mask_crop.shape
    offset = (int(0.7 * background_h - crop_h / 2), int(0.2 * background_w - crop_w / 2))
    offset = (
        int(np.clip(offset[0], 1, background_h - crop_h - 1)),
        int(np.clip(offset[1], 1, background_w - crop_w - 1)),
    )

    direct = direct_paste(background, src_crop, mask_crop, offset)
    poisson = poisson_blend(background, src_crop, mask_crop, offset, mixing=False)
    poisson_mix = poisson_blend(background, src_crop, mask_crop, offset, mixing=True)

    center = (offset[1] + crop_w // 2, offset[0] + crop_h // 2)
    clone = cv2.seamlessClone(
        cv2.cvtColor(src_crop, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(background, cv2.COLOR_RGB2BGR),
        mask_crop * 255,
        center,
        cv2.NORMAL_CLONE,
    )
    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)

    methods = [
        ('Ghép trực tiếp (baseline)', direct),
        ('Poisson (gradient nguồn)', poisson),
        ('Poisson (mixing gradients)', poisson_mix),
        ('cv2.seamlessClone', clone),
    ]
    metrics = pd.DataFrame([
        {
            'phương pháp': name,
            'seam_discontinuity': round(seam_discontinuity(result, mask_crop, offset), 3),
            'seam_edge_energy': round(seam_edge_energy(result, mask_crop, offset), 3),
            'object_color_shift': round(object_color_shift(result, src_crop, mask_crop, offset), 3),
        }
        for name, result in methods
    ])
    metrics.to_csv(output_dir / 'part1_gradient_metrics.csv', index=False)

    utils.show_grid(
        [('Ảnh nền', background), *methods],
        columns=3,
        figsize=(18, 9),
        save_name=output_dir / 'part1_comparison.png',
    )
    return metrics, mask_method


def run_part2(panorama_sets, output_dir, blend_mode='feather'):
    """Run SIFT and ORB for each set, saving panoramas and CSV summaries."""
    experiment_results = {}
    summary_rows = []
    pair_detail_rows = []

    for set_name, images in panorama_sets.items():
        if len(images) < 2:
            continue

        for method in ('SIFT', 'ORB'):
            panorama, summary, _ = stitch(images, method=method, blend_mode=blend_mode)
            experiment_results[(set_name, method)] = (panorama, summary)
            pair_metrics = summary.get('pair_metrics', []) if summary else []
            status = summary.get('status', 'FAIL') if summary else 'FAIL'

            summary_rows.append({
                'set': set_name,
                'method': method,
                'status': status,
                'n_images': len(images),
                'total_matches': sum(pair['matches'] for pair in pair_metrics),
                'total_inliers': sum(pair['inliers'] for pair in pair_metrics),
                'avg_inlier_ratio': round(float(np.mean([p['inlier_ratio'] for p in pair_metrics])), 3)
                if pair_metrics else float('nan'),
                'avg_reproj_error': round(float(np.nanmean([p['reproj_error'] for p in pair_metrics])), 3)
                if pair_metrics else float('nan'),
                'avg_photo_error': round(
                    float(np.nanmean([p.get('photo_error', float('nan')) for p in pair_metrics])), 3
                ) if pair_metrics else float('nan'),
                'time_s': summary.get('time', float('nan')) if summary else float('nan'),
            })

            if panorama is not None:
                save_rgb(output_dir / f'pano_{set_name}_{method.lower()}.jpg', panorama)

            for pair in pair_metrics:
                detail = dict(pair)
                detail.setdefault('photo_error', float('nan'))
                detail['set'] = set_name
                detail['method'] = method
                pair_detail_rows.append(detail)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'part2_experiment_summary.csv', index=False)
    if pair_detail_rows:
        columns = [
            'set', 'method', 'pair', 'matches', 'inliers',
            'inlier_ratio', 'reproj_error', 'photo_error',
        ]
        pd.DataFrame(pair_detail_rows)[columns].to_csv(
            output_dir / 'part2_pair_details.csv', index=False
        )
    return experiment_results, summary_df
