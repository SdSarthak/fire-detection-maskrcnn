"""Tests for the inference CLI's file handling and unicode-safe image IO."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import infer
from conftest import make_fire_image
from src.dataset import load_image
from src.predictor import FirePredictor

UNICODE_NAME = '火災_été'  # "fire disaster_ete"


class OverlayOnly:
    """Just enough of FirePredictor to exercise save_overlay without weights."""

    render_overlay = FirePredictor.render_overlay
    save_overlay = FirePredictor.save_overlay


def write_png(path: Path, image=None) -> Path:
    """Write a PNG without cv2.imwrite so unicode paths survive on Windows."""
    image = make_fire_image() if image is None else image
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path


# --------------------------------------------------------------------------- #
# Unicode-safe image IO
# --------------------------------------------------------------------------- #
def test_load_image_reads_a_non_ascii_filename(tmp_path):
    path = write_png(tmp_path / f'{UNICODE_NAME}.png')

    image = load_image(path)

    assert image.shape == (48, 64, 3)


def test_load_image_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / 'absent.png')


def test_load_image_reports_an_empty_file(tmp_path):
    path = tmp_path / 'empty.png'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='empty'):
        load_image(path)


def test_save_overlay_lands_at_the_requested_unicode_path(tmp_path):
    """cv2.imwrite returns True while writing to a mangled ANSI filename."""
    destination = tmp_path / f'prediction_{UNICODE_NAME}.png'
    image = make_fire_image()
    result = {'detections': [], 'masks': np.zeros((0, 48, 64), dtype=bool)}

    written = OverlayOnly().save_overlay(image, result, destination)

    assert written == destination
    assert destination.exists() and destination.stat().st_size > 0
    # Exactly one file: no mojibake twin left behind.
    assert [p.name for p in tmp_path.iterdir()] == [destination.name]
    assert load_image(destination).shape == (48, 64, 3)


def test_save_overlay_rejects_an_unencodable_extension(tmp_path):
    result = {'detections': [], 'masks': np.zeros((0, 48, 64), dtype=bool)}
    with pytest.raises((IOError, cv2.error)):
        OverlayOnly().save_overlay(make_fire_image(), result,
                                   tmp_path / 'overlay.xyz')


# --------------------------------------------------------------------------- #
# Overlay filename collisions
# --------------------------------------------------------------------------- #
def test_same_stem_with_different_extensions_gets_distinct_overlays(tmp_path):
    images = [tmp_path / 'fire.jpg', tmp_path / 'fire.png']

    destinations = infer.overlay_paths(images, tmp_path / 'out')

    assert len(set(destinations.values())) == 2


def test_same_basename_in_two_directories_gets_distinct_overlays(tmp_path):
    images = [tmp_path / 'a' / 'fire.jpg', tmp_path / 'b' / 'fire.jpg']

    destinations = infer.overlay_paths(images, tmp_path / 'out')

    assert len(set(destinations.values())) == 2
    assert all(p.parent == tmp_path / 'out' for p in destinations.values())


def test_distinct_stems_keep_their_natural_names(tmp_path):
    images = [tmp_path / 'a.jpg', tmp_path / 'b.jpg']

    destinations = infer.overlay_paths(images, tmp_path / 'out')

    assert destinations[images[0]].name == 'prediction_a.png'
    assert destinations[images[1]].name == 'prediction_b.png'


# --------------------------------------------------------------------------- #
# CLI validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('argv', [
    ['--score-threshold', '1.5'],
    ['--score-threshold', '-0.1'],
    ['--mask-threshold', '2'],
])
def test_out_of_range_thresholds_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        infer.parse_args(argv)

    assert excinfo.value.code == 2
    assert argv[0] in capsys.readouterr().err


def test_thresholds_inside_the_unit_interval_are_accepted():
    args = infer.parse_args(['--score-threshold', '0.4', '--mask-threshold', '0.6'])
    assert (args.score_threshold, args.mask_threshold) == (0.4, 0.6)


def test_missing_input_path_exits(tmp_path):
    with pytest.raises(SystemExit, match='Input not found'):
        infer.collect_inputs(tmp_path / 'nowhere')


def test_collect_inputs_accepts_a_single_file(tmp_path):
    path = write_png(tmp_path / 'one.png')
    assert infer.collect_inputs(path) == [path]


def test_missing_checkpoint_exits_with_training_instructions(tmp_path):
    with pytest.raises(SystemExit, match='Train one first'):
        infer.main(['--model', str(tmp_path / 'absent.pt'),
                    '--input', str(tmp_path)])


# --------------------------------------------------------------------------- #
# End-to-end run over a mixed directory
# --------------------------------------------------------------------------- #
def test_run_survives_a_corrupt_file_and_reports_it(tmp_path, monkeypatch):
    images_dir = tmp_path / 'images'
    write_png(images_dir / 'good.png')
    (images_dir / 'broken.png').write_bytes(b'not a png at all')

    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'placeholder')

    class Stub:
        def predict_file(self, path):
            if 'broken' in Path(path).name:
                raise ValueError('Failed to decode image')
            image = make_fire_image()
            return {'detections': [], 'num_detections': 0, 'is_fire': False,
                    'confidence': 0.0, 'fire_area_ratio': 0.0,
                    'masks': np.zeros((0, 48, 64), dtype=bool),
                    'filename': Path(path).name}, image

        def save_overlay(self, image, result, output_path):
            return write_png(Path(output_path), image)

    monkeypatch.setattr(infer.FirePredictor, 'from_checkpoint',
                        staticmethod(lambda *a, **k: Stub()))

    exit_code = infer.main(['--model', str(checkpoint),
                            '--input', str(images_dir),
                            '--output-dir', str(tmp_path / 'out'),
                            '--json', str(tmp_path / 'out.json')])

    assert exit_code == 0
    results = json.loads((tmp_path / 'out.json').read_text(encoding='utf-8'))
    by_name = {r['filename']: r for r in results}
    assert 'error' in by_name['broken.png']
    assert 'error' not in by_name['good.png']


def test_run_returns_non_zero_when_every_image_fails(tmp_path, monkeypatch):
    images_dir = tmp_path / 'images'
    write_png(images_dir / 'a.png')

    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'placeholder')

    class AlwaysFails:
        def predict_file(self, path):
            raise ValueError('nope')

    monkeypatch.setattr(infer.FirePredictor, 'from_checkpoint',
                        staticmethod(lambda *a, **k: AlwaysFails()))

    assert infer.main(['--model', str(checkpoint), '--input', str(images_dir),
                       '--output-dir', str(tmp_path / 'out'),
                       '--json', str(tmp_path / 'out.json')]) == 1
