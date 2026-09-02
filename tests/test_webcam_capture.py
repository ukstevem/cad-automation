"""
Tests for the pure device-identity logic in tools/webcam_capture.py.

No camera needed: these cover the naming that decides which photo belongs to which physical
camera. That mapping is load-bearing — ``fit_multiview.py --cam-profile SUBSTR=PATH`` matches on
the filename, so a wrong or colliding tag silently pairs a photo with the wrong intrinsics.
"""
import os
import sys

import pytest

pytest.importorskip("cv2")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import webcam_capture as WC  # noqa: E402


BY_ID = WC.BY_ID_DIR


def test_tag_uses_the_usb_serial_when_present():
    dev = f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920_9F8E7D6C-video-index0"
    assert WC.short_tag(dev, 0) == "9F8E7D6C"


def test_tag_falls_back_to_the_model_when_there_is_no_serial():
    dev = f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920-video-index0"
    assert WC.short_tag(dev, 0) == "C920"


def test_two_serialled_cameras_get_distinct_tags():
    devs = [f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920_AAAA1111-video-index0",
            f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920_BBBB2222-video-index0"]
    tags = WC.unique_tags(devs)
    assert tags[devs[0]] == "AAAA1111"
    assert tags[devs[1]] == "BBBB2222"


def test_identical_cameras_without_serials_do_not_collide():
    """
    Two of the same model reporting no serial derive the same raw tag. Both cameras share the
    shot's timestamp, so an unresolved collision would have one overwrite the other's photo.
    """
    devs = [f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920-video-index0",
            f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920-video-index0.2"]
    tags = WC.unique_tags(devs)
    assert len(set(tags.values())) == 2, f"tags collided: {tags}"


def test_tags_are_filename_safe():
    devs = [f"{BY_ID}/usb-046d_HD_Pro_Webcam_C920_9F8E:7D/6C-video-index0"]
    tag = WC.unique_tags(devs)[devs[0]]
    assert tag == "".join(ch for ch in tag if ch.isalnum())


def test_discover_devices_returns_a_list_when_nothing_is_attached(monkeypatch):
    monkeypatch.setattr(WC.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(WC.os.path, "exists", lambda p: False)
    assert WC.discover_devices() == []


def test_discover_devices_prefers_by_id_and_takes_only_index0(monkeypatch):
    entries = [
        "usb-046d_HD_Pro_Webcam_C920_AAAA-video-index0",
        "usb-046d_HD_Pro_Webcam_C920_AAAA-video-index1",   # metadata node, must be ignored
        "usb-046d_HD_Pro_Webcam_C920_BBBB-video-index0",
    ]
    monkeypatch.setattr(WC.os.path, "isdir", lambda p: p == WC.BY_ID_DIR)
    monkeypatch.setattr(WC.os, "listdir", lambda p: entries)
    found = WC.discover_devices()
    assert len(found) == 2
    assert all(f.endswith("-video-index0") for f in found)
