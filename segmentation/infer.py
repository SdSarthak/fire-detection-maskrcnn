"""
Inference pipeline for Mask R-CNN fire detection.

Runs a trained checkpoint over an image or a directory of images, writes
mask overlays to ``outputs/`` and prints/persists the detections as JSON.

    python infer.py --input data/test
    python infer.py --input photo.jpg --score-threshold 0.4 --json results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (OUTPUTS_DIR, TEST_DIR, FireDetectionConfig,
                        ensure_directories)
from src.dataset import list_images
from src.predictor import FirePredictor


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run fire segmentation on images with a trained model.')
    parser.add_argument('--model', default=None,
                        help='Checkpoint path (default: weights/fire_detection_model.pt)')
    parser.add_argument('--input', default=str(TEST_DIR),
                        help='Image file or directory of images')
    parser.add_argument('--output-dir', default=str(OUTPUTS_DIR),
                        help='Where overlay images are written')
    parser.add_argument('--json', default=None,
                        help='Optional path for the aggregated JSON results')
    parser.add_argument('--score-threshold', type=float, default=None,
                        help='Minimum detection confidence (default: from config)')
    parser.add_argument('--mask-threshold', type=float, default=None,
                        help='Probability cut-off used to binarise masks')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--no-overlay', action='store_true',
                        help='Skip writing the annotated images')
    return parser.parse_args(argv)


def collect_inputs(target: Path):
    """Resolve the --input argument to a concrete list of image paths."""
    if target.is_dir():
        return list_images(target)
    if target.is_file():
        return [target]
    raise SystemExit(f'Input not found: {target}')


def main(argv=None) -> int:
    args = parse_args(argv)
    ensure_directories()

    config = FireDetectionConfig.from_env()
    model_path = Path(args.model or config.MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(
            f'Model checkpoint not found: {model_path}\n'
            'Train one first:  python train.py')

    images = collect_inputs(Path(args.input))
    if not images:
        raise SystemExit(f'No supported images found in {args.input}')

    print('=' * 60)
    print('Mask R-CNN fire detection - inference')
    print('=' * 60)
    print(f'Model:  {model_path}')
    print(f'Images: {len(images)}')

    predictor = FirePredictor.from_checkpoint(
        model_path, device=args.device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold)

    output_dir = Path(args.output_dir)
    results = []

    for index, image_path in enumerate(images, start=1):
        try:
            result, image = predictor.predict_file(image_path)
        except Exception as exc:  # a single unreadable file must not stop the run
            print(f'  [{index}/{len(images)}] {image_path.name}: failed - {exc}')
            results.append({'filename': image_path.name, 'error': str(exc)})
            continue

        verdict = 'FIRE' if result['is_fire'] else 'no fire'
        print(f'  [{index}/{len(images)}] {image_path.name}: {verdict} '
              f'({result["num_detections"]} instance(s), '
              f'confidence {result["confidence"]:.3f}, '
              f'area {result["fire_area_ratio"] * 100:.2f}%)')

        if not args.no_overlay:
            overlay_path = output_dir / f'prediction_{image_path.stem}.png'
            predictor.save_overlay(image, result, overlay_path)
            result['overlay_path'] = str(overlay_path)

        results.append(FirePredictor.serializable(result))

    json_path = Path(args.json) if args.json else output_dir / 'predictions.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    fires = sum(1 for r in results if r.get('is_fire'))
    print('\n' + '=' * 60)
    print(f'Done: fire detected in {fires}/{len(results)} image(s)')
    print(f'Results: {json_path}')
    if not args.no_overlay:
        print(f'Overlays: {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
