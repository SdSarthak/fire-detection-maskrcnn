"""Tests for VIA annotation parsing and polygon rasterisation."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import rectangle_polygon, via_project
from src.dataset import (VIA_METADATA_KEY, drop_degenerate, load_via_annotations,
                         masks_to_boxes, parse_via_annotations, polygon_to_mask,
                         polygons_to_masks)


def test_parses_flat_via_export():
    data = via_project({'fire.jpg': [rectangle_polygon(2, 3, 10, 12)]})
    parsed = parse_via_annotations(data)

    assert list(parsed) == ['fire.jpg']
    assert len(parsed['fire.jpg']) == 1
    assert parsed['fire.jpg'][0].shape == (4, 2)


def test_parses_nested_via_img_metadata():
    """Newer VIA versions wrap the records under _via_img_metadata."""
    inner = via_project({'fire.jpg': [rectangle_polygon(0, 0, 5, 5)]})
    parsed = parse_via_annotations({'_via_settings': {}, VIA_METADATA_KEY: inner})

    assert 'fire.jpg' in parsed
    assert len(parsed['fire.jpg']) == 1


def test_parses_regions_stored_as_dict():
    """VIA also serialises regions as an index-keyed object."""
    data = {
        'fire.jpg1': {
            'filename': 'fire.jpg',
            'regions': {
                '0': {'shape_attributes': rectangle_polygon(0, 0, 4, 4),
                      'region_attributes': {'object': 'fire'}},
                '1': {'shape_attributes': rectangle_polygon(5, 5, 9, 9),
                      'region_attributes': {'object': 'fire'}},
            },
        }
    }
    assert len(parse_via_annotations(data)['fire.jpg']) == 2


@pytest.mark.parametrize('shape, expected_points', [
    ({'name': 'rect', 'x': 1, 'y': 2, 'width': 6, 'height': 8}, 4),
    ({'name': 'circle', 'cx': 10, 'cy': 10, 'r': 5}, 32),
    ({'name': 'ellipse', 'cx': 10, 'cy': 10, 'rx': 6, 'ry': 3}, 32),
])
def test_non_polygon_shapes_are_converted(shape, expected_points):
    data = {'img1': {'filename': 'img.jpg',
                     'regions': [{'shape_attributes': shape, 'region_attributes': {}}]}}
    polygons = parse_via_annotations(data)['img.jpg']

    assert len(polygons) == 1
    assert polygons[0].shape == (expected_points, 2)


@pytest.mark.parametrize('shape', [
    {'name': 'polygon', 'all_points_x': [1, 2], 'all_points_y': [1, 2]},   # too few
    {'name': 'rect', 'x': 0, 'y': 0, 'width': 0, 'height': 5},             # zero area
    {'name': 'point', 'cx': 3, 'cy': 4},                                   # no area
])
def test_unusable_shapes_are_dropped(shape):
    data = {'img1': {'filename': 'img.jpg',
                     'regions': [{'shape_attributes': shape, 'region_attributes': {}}]}}
    assert parse_via_annotations(data)['img.jpg'] == []


def test_class_filter_selects_matching_regions_only():
    data = {
        'img1': {
            'filename': 'img.jpg',
            'regions': [
                {'shape_attributes': rectangle_polygon(0, 0, 4, 4),
                 'region_attributes': {'object': 'fire'}},
                {'shape_attributes': rectangle_polygon(5, 5, 9, 9),
                 'region_attributes': {'object': 'smoke'}},
            ],
        }
    }
    assert len(parse_via_annotations(data, class_filter=['fire'])['img.jpg']) == 1
    assert len(parse_via_annotations(data, class_filter=['FIRE'])['img.jpg']) == 1
    assert len(parse_via_annotations(data)['img.jpg']) == 2


def test_load_via_annotations_reads_file(tmp_path):
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(via_project({'a.jpg': [rectangle_polygon(0, 0, 6, 6)]})),
                    encoding='utf-8')

    assert 'a.jpg' in load_via_annotations(path)


def test_load_via_annotations_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_via_annotations(tmp_path / 'nope.json')


def test_polygon_to_mask_fills_the_region():
    polygon = np.array([[2, 2], [8, 2], [8, 6], [2, 6]], dtype=np.float64)
    mask = polygon_to_mask(polygon, height=10, width=10)

    assert mask.dtype == np.uint8
    assert mask[3, 4] == 1
    assert mask[0, 0] == 0
    # fillPoly includes both edges: rows 2..6 x cols 2..8
    assert mask.sum() == 35
    assert mask[7, 4] == 0 and mask[3, 9] == 0


def test_polygon_points_outside_the_image_are_clipped():
    polygon = np.array([[-50, -50], [100, -50], [100, 100], [-50, 100]], dtype=np.float64)
    mask = polygon_to_mask(polygon, height=8, width=8)

    assert mask.sum() == 64


def test_masks_to_boxes_are_tight():
    masks = np.zeros((1, 20, 20), dtype=np.uint8)
    masks[0, 5:12, 3:9] = 1

    np.testing.assert_allclose(masks_to_boxes(masks)[0], [3, 5, 9, 12])


def test_masks_to_boxes_handles_empty_stack():
    assert masks_to_boxes(np.zeros((0, 5, 5), dtype=np.uint8)).shape == (0, 4)


def test_drop_degenerate_removes_empty_instances():
    masks = np.zeros((2, 10, 10), dtype=np.uint8)
    masks[0, 2:8, 2:8] = 1  # kept
    boxes = masks_to_boxes(masks)

    kept_masks, kept_boxes = drop_degenerate(masks, boxes)

    assert kept_masks.shape[0] == 1
    assert kept_boxes.shape == (1, 4)


def test_polygons_to_masks_stacks_instances():
    polygons = [np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64),
                np.array([[5, 5], [9, 5], [9, 9], [5, 9]], dtype=np.float64)]

    masks = polygons_to_masks(polygons, height=10, width=10)

    assert masks.shape == (2, 10, 10)
    assert not np.logical_and(masks[0], masks[1]).any()


# --------------------------------------------------------------------------- #
# Out-of-frame polygons and malformed annotations (Pass 2)
# --------------------------------------------------------------------------- #
def test_polygon_extending_past_the_frame_is_clipped_geometrically():
    """Snapping vertices to the border would shrink the visible area.

    The triangle below has one vertex far to the left of the image. Clamping
    that vertex onto x=0 changes the slope of both edges that leave the frame
    and loses roughly 40% of the pixels that are genuinely inside it.
    """
    triangle = np.array([[-30.0, 10.0], [10.0, 0.0], [10.0, 20.0]])

    mask = polygon_to_mask(triangle, height=20, width=20)

    snapped = np.clip(triangle, [0, 0], [19, 19])
    naive = polygon_to_mask(snapped, height=20, width=20)

    assert mask.sum() > naive.sum()
    assert mask.sum() == pytest.approx(196, abs=8)
    # At x=0 the true triangle spans y in [2.5, 17.5]; the snapped one is a
    # single point there, because its apex was dragged onto the border.
    assert mask[:, 0].sum() == pytest.approx(15, abs=2)
    assert naive[:, 0].sum() <= 1


def test_polygon_entirely_outside_the_frame_rasterises_empty():
    outside = np.array([[-50.0, -50.0], [-40.0, -50.0], [-40.0, -40.0]])
    assert polygon_to_mask(outside, height=20, width=20).sum() == 0


def test_polygon_covering_the_whole_frame_fills_it():
    cover = np.array([[-5.0, -5.0], [25.0, -5.0], [25.0, 25.0], [-5.0, 25.0]])
    assert polygon_to_mask(cover, height=20, width=20).all()


def test_polygon_with_non_finite_coordinates_is_dropped_not_raised():
    broken = np.array([[0.0, 0.0], [np.nan, 5.0], [5.0, 5.0]])
    assert polygon_to_mask(broken, height=10, width=10).sum() == 0

    infinite = np.array([[0.0, 0.0], [np.inf, 5.0], [5.0, 5.0]])
    assert polygon_to_mask(infinite, height=10, width=10).sum() == 0


def test_polygon_to_mask_rejects_non_positive_canvas():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]])
    with pytest.raises(ValueError, match='positive'):
        polygon_to_mask(square, height=0, width=10)


def test_parse_rejects_non_object_json():
    with pytest.raises(TypeError, match='JSON object'):
        parse_via_annotations([{'filename': 'a.jpg'}])


def test_parse_rejects_non_object_metadata_wrapper():
    with pytest.raises(TypeError, match=VIA_METADATA_KEY):
        parse_via_annotations({VIA_METADATA_KEY: ['not', 'a', 'dict']})


def test_non_numeric_polygon_points_are_skipped_not_fatal():
    data = via_project({'fire.jpg': [{'name': 'polygon',
                                      'all_points_x': ['a', 'b', 'c'],
                                      'all_points_y': [1, 2, 3]}]})
    assert parse_via_annotations(data) == {'fire.jpg': []}


def test_non_numeric_rect_attributes_are_skipped():
    data = via_project({'fire.jpg': [{'name': 'rect', 'x': 'left', 'y': 0,
                                      'width': 5, 'height': 5}]})
    assert parse_via_annotations(data) == {'fire.jpg': []}


def test_load_via_annotations_reports_malformed_json_with_position(tmp_path):
    path = tmp_path / 'broken.json'
    path.write_text('{"a": ', encoding='utf-8')

    with pytest.raises(ValueError) as excinfo:
        load_via_annotations(path)

    message = str(excinfo.value)
    assert 'broken.json' in message and 'not valid JSON' in message


def test_load_via_annotations_handles_unicode_filenames(tmp_path):
    path = tmp_path / 'unicode.json'
    payload = via_project({'\u706b\u707d_\u00e9t\u00e9.jpg': [rectangle_polygon(1, 1, 5, 5)]})
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    parsed = load_via_annotations(path)

    assert list(parsed) == ['\u706b\u707d_\u00e9t\u00e9.jpg']
