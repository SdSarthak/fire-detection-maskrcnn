"""
Training pipeline for Mask R-CNN fire detection.

Fine-tunes a COCO-pretrained Mask R-CNN on fire images annotated with the
VGG Image Annotator, evaluating each epoch and keeping the best checkpoint.

    python train.py --epochs 30
    python train.py --data-dir /data/fire --epochs 10 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (ANNOTATIONS_DIR, DATA_DIR, OUTPUTS_DIR, WEIGHTS_DIR,
                        FireDetectionConfig, ensure_directories)
from src.dataset import FireSegmentationDataset, collate_fn
from src.model import (build_model, compute_validation_loss, create_optimizer,
                       create_scheduler, evaluate, resolve_device,
                       save_checkpoint, train_one_epoch)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the Mask R-CNN fire detection model.')
    parser.add_argument('--data-dir', default=str(DATA_DIR),
                        help='Root data directory containing train/ and val/')
    parser.add_argument('--annotations-dir', default=str(ANNOTATIONS_DIR),
                        help='Directory holding the VIA annotation JSON files')
    parser.add_argument('--train-annotations', default='train_annotations.json',
                        help='VIA annotation file for the training split')
    parser.add_argument('--val-annotations', default='val_annotations.json',
                        help='VIA annotation file for the validation split')
    parser.add_argument('--class-filter', default='fire',
                        help='Comma separated region attribute values to keep '
                             '(empty string keeps every region)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None, dest='learning_rate')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', default=None, choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--pretrained', default=None,
                        choices=['coco', 'imagenet', 'none'],
                        help='Which pre-trained weights to start from')
    parser.add_argument('--output', default=None,
                        help='Checkpoint path (default: weights/fire_detection_model.pt)')
    parser.add_argument('--history', default=str(OUTPUTS_DIR / 'training_history.json'),
                        help='Where to write the per-epoch metrics')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> FireDetectionConfig:
    """Layer CLI flags on top of the environment-derived configuration."""
    return FireDetectionConfig.from_env(
        TRAIN_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.learning_rate,
        NUM_WORKERS=args.num_workers,
        DEVICE=args.device,
        PRETRAINED_WEIGHTS=args.pretrained,
        MODEL_PATH=args.output,
        SEED=args.seed,
    )


def build_dataloaders(args: argparse.Namespace, config: FireDetectionConfig):
    """Create the train and validation loaders, tolerating a missing val split."""
    data_dir = Path(args.data_dir)
    annotations_dir = Path(args.annotations_dir)
    class_filter = [c.strip() for c in args.class_filter.split(',') if c.strip()] or None

    train_dataset = FireSegmentationDataset(
        images_dir=data_dir / 'train',
        annotations=annotations_dir / args.train_annotations,
        class_filter=class_filter,
        horizontal_flip_prob=config.HORIZONTAL_FLIP_PROB,
    )
    if len(train_dataset) == 0:
        raise SystemExit(
            f'No annotated training images found.\n'
            f'  images:      {data_dir / "train"}\n'
            f'  annotations: {annotations_dir / args.train_annotations}\n'
            'Annotate fire regions with the VGG Image Annotator and export the '
            'project as JSON, then re-run.')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS,
                              collate_fn=collate_fn)

    val_annotations = annotations_dir / args.val_annotations
    val_images = data_dir / 'val'
    val_loader = None
    if val_images.exists() and val_annotations.exists():
        val_dataset = FireSegmentationDataset(
            images_dir=val_images,
            annotations=val_annotations,
            class_filter=class_filter,
            horizontal_flip_prob=0.0,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=config.NUM_WORKERS,
                                    collate_fn=collate_fn)

    return train_loader, val_loader


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    ensure_directories()
    torch.manual_seed(config.SEED)

    print('=' * 60)
    print('Mask R-CNN fire detection - training')
    print('=' * 60)
    config.display()

    device = resolve_device(config.DEVICE)
    print(f'\nDevice: {device}')

    train_loader, val_loader = build_dataloaders(args, config)
    print(f'Training images:   {len(train_loader.dataset)}')
    print(f'Validation images: {len(val_loader.dataset) if val_loader else 0}')
    if val_loader is None:
        print('  (no validation split found - the last epoch will be kept)')

    skipped = train_loader.dataset.unannotated
    if skipped:
        print(f'  skipped {len(skipped)} unannotated image(s): '
              f'{", ".join(skipped[:5])}{" ..." if len(skipped) > 5 else ""}')

    print('\nBuilding model...')
    model = build_model(config)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable parameters: {trainable:,}')

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    checkpoint_path = Path(config.MODEL_PATH)
    best_score = -1.0
    history = []

    print(f'\nTraining for {config.TRAIN_EPOCHS} epoch(s)...')
    for epoch in range(config.TRAIN_EPOCHS):
        losses = train_one_epoch(model, optimizer, train_loader, device, epoch, config)
        scheduler.step()

        record = {'epoch': epoch, **{k: round(v, 5) for k, v in losses.items()}}

        if val_loader is not None:
            record['val_loss'] = round(compute_validation_loss(model, val_loader, device), 5)
            metrics = evaluate(model, val_loader, device, config)
            record.update({f'val_{k}': round(v, 5) for k, v in metrics.items()})
            score = metrics['f1']
        else:
            score = -losses['loss']  # no validation data: track training loss

        history.append(record)
        summary = ' '.join(f'{k}={v}' for k, v in record.items()
                           if k in {'loss', 'val_loss', 'val_f1', 'val_mean_iou'})
        print(f'  epoch {epoch} done in {losses["seconds"]:.1f}s | {summary}')

        if score > best_score:
            best_score = score
            save_checkpoint(model, checkpoint_path, config, epoch=epoch,
                            metrics={k: v for k, v in record.items()
                                     if isinstance(v, (int, float))})
            print(f'  new best checkpoint -> {checkpoint_path}')

    if best_score < 0 and not checkpoint_path.exists():
        save_checkpoint(model, checkpoint_path, config, epoch=config.TRAIN_EPOCHS - 1)

    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')

    print('\n' + '=' * 60)
    print('Training complete')
    print('=' * 60)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'History:    {history_path}')
    print(f'Weights dir: {WEIGHTS_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
