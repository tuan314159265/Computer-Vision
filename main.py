#!/usr/bin/env python3
"""Command-line entry point for the CO3057 Computer Vision assignment."""
import argparse
import time
from pathlib import Path

from lib.data import load_project_inputs
from lib.experiments import run_part1, run_part2


def parse_args():
    parser = argparse.ArgumentParser(
        description='CO3057 BTL2: Gradient Domain Editing and Panorama Stitching'
    )
    parser.add_argument('--data-dir', default='data', help='Input data directory (default: data)')
    parser.add_argument('--output-dir', default='outputs', help='Results directory (default: outputs)')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    background, source, mask_path, panorama_sets, input_status = load_project_inputs(
        data_dir, output_dir
    )
    print(f"[gradient] {input_status['gradient']}")
    for set_name, status in input_status.items():
        if set_name != 'gradient':
            print(f'[{set_name}] {status} ({len(panorama_sets[set_name])} ảnh)')

    print('\nChạy Phần 1 — Gradient Domain Editing...')
    metrics, mask_method = run_part1(background, source, mask_path, output_dir)
    print(f'Phần 1 OK | mask: {mask_method}')
    print(metrics.to_string(index=False))

    print('\nChạy Phần 2 — SIFT & ORB panorama...')
    start_time = time.time()
    experiment_results, summary = run_part2(panorama_sets, output_dir)
    print(summary.to_string(index=False))
    print(f'Phần 2 xong trong {time.time() - start_time:.1f}s')

    print('\n=== KẾT QUẢ ===')
    for (set_name, method), (panorama, result) in sorted(experiment_results.items()):
        if panorama is not None:
            print(f'✓ {set_name}/{method}: {panorama.shape[1]}x{panorama.shape[0]} px  {result["status"]}')
        else:
            print(f'✗ {set_name}/{method}: FAIL')

    print(f'\nCác file đã xuất ra {output_dir}:')
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f' - {path.name}')


if __name__ == '__main__':
    main()
