#!/usr/bin/env python3
"""Runner đầy đủ cho CO3057 BTL2 (tách từ notebook).

Chạy:
    python main.py [--data-dir data] [--output-dir outputs]

Nếu thiếu ảnh thật, tự sinh ảnh DEMO (có nhãn rõ) để kiểm tra kỹ thuật.
Kết quả DEMO không dùng để nộp.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lib import utils
from lib.gradient_domain import (
    build_mask,
    crop_to_mask,
    direct_paste,
    list_images,
    load_rgb,
    object_color_shift,
    poisson_blend,
    save_rgb,
    seam_discontinuity,
    seam_edge_energy,
)
from lib.panorama import stitch


def run_part1(background, source, mask_path, output_dir):
    """Gradient Domain Editing: baseline vs Poisson (2 biến thể) vs seamlessClone."""
    mask, mask_method = build_mask(source, mode='auto', mask_path=mask_path)

    src_crop, mask_crop = crop_to_mask(source, mask)
    max_h, max_w = int(0.45 * background.shape[0]), int(0.45 * background.shape[1])
    s = min(1.0, max_h / src_crop.shape[0], max_w / src_crop.shape[1])
    if s < 1.0:
        src_crop = cv2.resize(src_crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask_crop = cv2.resize(mask_crop, (src_crop.shape[1], src_crop.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_crop = (mask_crop > 127).astype(np.uint8)

    bh, bw = background.shape[:2]
    ch, cw = mask_crop.shape
    offset = (int(0.7 * bh - ch / 2), int(0.2 * bw - cw / 2))
    offset = (int(np.clip(offset[0], 1, bh - ch - 1)),
              int(np.clip(offset[1], 1, bw - cw - 1)))

    direct = direct_paste(background, src_crop, mask_crop, offset)
    poisson = poisson_blend(background, src_crop, mask_crop, offset, mixing=False)
    poisson_mix = poisson_blend(background, src_crop, mask_crop, offset, mixing=True)

    center = (offset[1] + cw // 2, offset[0] + ch // 2)
    clone = cv2.seamlessClone(cv2.cvtColor(src_crop, cv2.COLOR_RGB2BGR),
                              cv2.cvtColor(background, cv2.COLOR_RGB2BGR),
                              mask_crop * 255, center, cv2.NORMAL_CLONE)
    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)

    rows = []
    for name, res in [('Ghép trực tiếp (baseline)', direct),
                      ('Poisson (gradient nguồn)', poisson),
                      ('Poisson (mixing gradients)', poisson_mix),
                      ('cv2.seamlessClone', clone)]:
        rows.append({
            'phương pháp': name,
            'seam_discontinuity': round(seam_discontinuity(res, mask_crop, offset), 3),
            'seam_edge_energy': round(seam_edge_energy(res, mask_crop, offset), 3),
            'object_color_shift': round(object_color_shift(res, src_crop, mask_crop, offset), 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'part1_gradient_metrics.csv', index=False)

    utils.show_grid([('Ảnh nền', background),
                     ('Baseline — ghép trực tiếp', direct),
                     ('Poisson (gradient nguồn)', poisson),
                     ('Poisson (mixing gradients)', poisson_mix),
                     ('cv2.seamlessClone', clone)],
                    columns=3, figsize=(18, 9),
                    save_name=output_dir / 'part1_comparison.png')
    return df, mask_method


def load_pano_set(folder, demo_seed):
    """Nạp ảnh một bộ panorama; nếu thiếu dữ liệu thật thì sinh DEMO dự phòng."""
    files = list_images(folder)
    if len(files) >= 2:
        return [load_rgb(f) for f in files], 'REAL'
    return utils.make_demo_panorama(seed=demo_seed, n_images=4), 'DEMO'


def run_part2(pano_sets, output_dir):
    """Chạy SIFT & ORB trên mọi bộ ảnh, ghi panorama + bảng số liệu."""
    blend = 'feather'
    experiment_results = {}
    summary_rows = []
    pair_detail_rows = []

    for set_name, imgs in pano_sets.items():
        if len(imgs) < 2:
            continue
        for method in ('SIFT', 'ORB'):
            pano, summary, _ = stitch(imgs, method=method, blend_mode=blend)
            experiment_results[(set_name, method)] = (pano, summary)

            if summary is None:
                status, t, pm = 'FAIL', float('nan'), []
            else:
                status, t, pm = summary['status'], summary['time'], summary['pair_metrics']

            summary_rows.append({
                'set': set_name,
                'method': method,
                'status': status,
                'n_images': len(imgs),
                'total_matches': sum(p['matches'] for p in pm),
                'total_inliers': sum(p['inliers'] for p in pm),
                'avg_inlier_ratio': round(float(np.mean([p['inlier_ratio'] for p in pm])), 3) if pm else float('nan'),
                'avg_reproj_error': round(float(np.nanmean([p['reproj_error'] for p in pm])), 3) if pm else float('nan'),
                'avg_photo_error': round(float(np.nanmean([p.get('photo_error', float('nan')) for p in pm])), 3) if pm else float('nan'),
                'time_s': t,
            })

            if pano is not None:
                save_rgb(output_dir / f'pano_{set_name}_{method.lower()}.jpg', pano)
            for pm in summary['pair_metrics'] if summary else []:
                d = dict(pm)
                d.setdefault('photo_error', float('nan'))
                d['set'] = set_name
                d['method'] = method
                pair_detail_rows.append(d)

    experiment_df = pd.DataFrame(summary_rows)
    experiment_df.to_csv(output_dir / 'part2_experiment_summary.csv', index=False)
    if pair_detail_rows:
        cols = ['set', 'method', 'pair', 'matches', 'inliers',
                'inlier_ratio', 'reproj_error', 'photo_error']
        pd.DataFrame(pair_detail_rows)[cols].to_csv(output_dir / 'part2_pair_details.csv', index=False)
    return experiment_results, experiment_df


def parse_args():
    p = argparse.ArgumentParser(description='CO3057 BTL2: Gradient Editing + Panorama')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', default='outputs')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phần 1: Gradient Domain Editing ----
    gradient_dir = data_dir / 'gradient'
    bg_path = gradient_dir / 'background.jpg'
    src_path = gradient_dir / 'source.jpg'
    mask_path = gradient_dir / 'mask.png'

    if bg_path.exists() and src_path.exists():
        background = load_rgb(bg_path)
        source = load_rgb(src_path, max_side=500)
        mask_file = mask_path if mask_path.exists() else None
        print('✓ Đã tìm thấy ảnh Gradient Domain (REAL).')
    else:
        demo_bg, demo_src, demo_mask = utils.make_demo_gradient_pair()
        background, source = demo_bg, demo_src
        mask_file = output_dir / 'demo_mask.png'
        cv2.imwrite(str(mask_file), demo_mask)
        print('⚠️ Thiếu ảnh Gradient Domain → dùng ảnh DEMO.')

    df1, mask_method = run_part1(background, source, mask_file, output_dir)
    print(f'Phần 1 OK | mask: {mask_method}')
    print(df1.to_string(index=False))

    # ---- Phần 2: Panorama Stitching ----
    pano_dirs = {'pano_set1': data_dir / 'pano_set1',
                 'pano_set2': data_dir / 'pano_set2'}
    pano_sets = {}
    for i, (name, folder) in enumerate(pano_dirs.items()):
        imgs, status = load_pano_set(folder, demo_seed=11 + i * 11)
        pano_sets[name] = imgs
        print(f'[{name}] {status} ({len(imgs)} ảnh)')

    print('\nChạy SIFT & ORB...')
    t0 = time.time()
    experiment_results, df2 = run_part2(pano_sets, output_dir)
    print(df2.to_string(index=False))
    print(f'\nPhần 2 xong trong {time.time() - t0:.1f}s')

    print('\n=== KẾT QUẢ ===')
    for (set_name, method), (pano, summary) in sorted(experiment_results.items()):
        if pano is not None:
            print(f'✓ {set_name}/{method}: {pano.shape[1]}x{pano.shape[0]} px  {summary["status"]}')
        else:
            print(f'✗ {set_name}/{method}: FAIL')

    print(f'\nCác file đã xuất ra {output_dir}:')
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(' -', f.name)


if __name__ == '__main__':
    main()
