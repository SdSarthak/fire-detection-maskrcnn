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
import math
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (ANNOTATIONS_DIR, DATA_DIR, OUTPUTS_DIR, WEIGHTS_DIR,
                        FireDetectionConfig, ensure_directories)
from src.dataset import FireSegmentationDataset, collate_fn
from src.model import (build_model, compute_validation_loss, create_optimizer,
                       create_scheduler, evaluate, resolve_device,
                       save_checkpoint, seed_worker, set_seed, train_one_epoch)


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
    parser.add_argument('--deterministic', action='store_true',
                        help='Force deterministic cuDNN kernels (slower, but '
                             'makes GPU runs reproducible)')
    args = parser.parse_args(argv)

    # argparse happily accepts --epochs 0 or --lr -1; catch them here so the
    # failure names the flag instead of surfacing deep inside the optimiser.
    for flag, value, minimum in (('--epochs', args.epochs, 1),
                                 ('--batch-size', args.batch_size, 1),
                                 ('--num-workers', args.num_workers, 0)):
        if value is not None and value < minimum:
            parser.error(f'{flag} must be >= {minimum}; got {value}')
    if args.learning_rate is not None and not args.learning_rate > 0:
        parser.error(f'--lr must be > 0; got {args.learning_rate}')
    if args.seed is not None and not 0 <= args.seed < 2 ** 32:
        parser.error(f'--seed must fit in 32 bits; got {args.seed}')
    return args


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


def build_dataloaders(args: argparse.Namespace, config: FireDetectionConfig,
                      generator=None):
    """Create the train and validation loaders, tolerating a missing val split."""
    data_dir = Path(args.data_dir)
    annotations_dir = Path(args.annotations_dir)
    if not data_dir.exists():
        raise SystemExit(f'Data directory does not exist: {data_dir}')
    if not annotations_dir.exists():
        raise SystemExit(f'Annotations directory does not exist: {annotations_dir}')
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
                              collate_fn=collate_fn, generator=generator,
                              worker_init_fn=seed_worker)

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
            leaked = check_split_leakage(train_dataset, val_dataset)
            if leaked:
                raise SystemExit(
                    f'{len(leaked)} image(s) appear in both the train and val '
                    f'splits ({", ".join(leaked[:5])}'
                    f'{" ..." if len(leaked) > 5 else ""}). Validation scores '
                    'would be measured on training data. Remove the duplicates '
                    'or point --data-dir at a properly split dataset.')
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=config.NUM_WORKERS,
                                    collate_fn=collate_fn)

    return train_loader, val_loader


def check_split_leakage(train_dataset, val_dataset) -> list:
    """Return image basenames present in both splits.

    Pointing ``data/train`` and ``data/val`` at overlapping folders silently
    turns every reported val_f1 into a training score.
    """
    train_names = {p.name for p in train_dataset.image_paths}
    val_names = {p.name for p in val_dataset.image_paths}
    return sorted(train_names & val_names)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    ensure_directories()
    generator = set_seed(config.SEED, deterministic=args.deterministic)

    print('=' * 60)
    print('Mask R-CNN fire detection - training')
    print('=' * 60)
    config.display()

    device = resolve_device(config.DEVICE)
    print(f'\nDevice: {device}')

    train_loader, val_loader = build_dataloaders(args, config, generator=generator)
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
    # Must be -inf, not -1.0: without a validation split the score is the
    # negated training loss, so any loss above 1.0 never beats -1.0 and the run
    # would finish having saved nothing (or, worse, silently leave a stale
    # checkpoint from an earlier run in place).
    best_score = float('-inf')
    best_epoch = -1
    history = []
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)

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
        # Flush every epoch: a crash (or a pre-emptible VM) at epoch 27 of 30
        # used to lose the whole history file.
        write_history(history_path, history)

        summary = ' '.join(f'{k}={v}' for k, v in record.items()
                           if k in {'loss', 'val_loss', 'val_f1', 'val_mean_iou'})
        print(f'  epoch {epoch} done in {losses["seconds"]:.1f}s | {summary}')

        if math.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(model, checkpoint_path, config, epoch=epoch,
                            metrics={k: v for k, v in record.items()
                                     if isinstance(v, (int, float))})
            print(f'  new best checkpoint -> {checkpoint_path}')

    if best_epoch < 0:
        # Every epoch scored non-finite (or there were none); still persist the
        # weights so the run is not a total loss.
        save_checkpoint(model, checkpoint_path, config,
                        epoch=max(config.TRAIN_EPOCHS - 1, 0))
        print(f'  no epoch produced a finite score; saved the final weights '
              f'-> {checkpoint_path}')

    print('\n' + '=' * 60)
    print('Training complete')
    print('=' * 60)
    print(f'Checkpoint: {checkpoint_path} (epoch {max(best_epoch, 0)})')
    print(f'History:    {history_path}')
    print(f'Weights dir: {WEIGHTS_DIR}')
    return 0


def write_history(path: Path, history) -> None:
    """Write the per-epoch metrics atomically so a crash cannot truncate them."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(history, indent=2), encoding='utf-8')
    os.replace(temporary, path)


if __name__ == '__main__':
    raise SystemExit(main())
