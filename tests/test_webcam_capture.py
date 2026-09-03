"""
Tests for the pure device-identity logic in tools/webcam_capture.py.

No camera needed: these cover the naming that decides which photo belongs to which physical
camera. That mapping is load-bearing — ``fit_multiview.py --cam-profile SUBSTR=PATH`` matches on
the filename, so a wrong or colliding tag silently pairs a photo with the wrong intrinsics.
"""
import os
import struct
import sys

import pytest

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


class _FakeProc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def test_get_ctrl_parses_menu_controls_with_a_trailing_label(monkeypatch):
    """
    v4l2-ctl labels menu controls: `auto_exposure: 1 (Manual Mode)`.

    Parsing the whole remainder returns None for exactly the mode controls that decide whether
    the manual values apply at all — which silently drops them from both the apply ordering and
    the verification. This was a live bug: auto_exposure was never actually being verified.
    """
    monkeypatch.setattr(WC, "_v4l2", lambda d, a: _FakeProc("auto_exposure: 1 (Manual Mode)\n"))
    assert WC.get_ctrl("/dev/video0", "auto_exposure") == 1


def test_get_ctrl_parses_a_plain_integer(monkeypatch):
    monkeypatch.setattr(WC, "_v4l2", lambda d, a: _FakeProc("exposure_time_absolute: 77\n"))
    assert WC.get_ctrl("/dev/video0", "exposure_time_absolute") == 77


def test_get_ctrl_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(WC, "_v4l2", lambda d, a: _FakeProc("", returncode=1))
    assert WC.get_ctrl("/dev/video0", "nope") is None


def test_apply_controls_sets_auto_toggles_before_the_values_they_gate(monkeypatch):
    """
    Order is load-bearing. Setting exposure_time_absolute while auto_exposure is still on fails
    with Permission denied and leaves the auto-chosen value; switching to manual afterwards then
    freezes it. Measured on the rig: value-then-manual kept 77, manual-then-value gave 8.
    """
    calls = []
    monkeypatch.setattr(WC, "set_ctrl", lambda d, n, v: calls.append(n) or True)
    monkeypatch.setattr(WC.time, "sleep", lambda s: None)
    # Deliberately hostile ordering: the value first, the mode toggle last.
    WC.apply_controls("/dev/video0", {
        "exposure_time_absolute": 8,
        "white_balance_temperature": 3696,
        "auto_exposure": 1,
        "white_balance_automatic": 0,
    })
    assert calls.index("auto_exposure") < calls.index("exposure_time_absolute")
    assert calls.index("white_balance_automatic") < calls.index("white_balance_temperature")


def test_geometry_critical_covers_focus_and_zoom_only():
    """Only focus and zoom change the camera model; exposure/WB change appearance."""
    assert "focus_absolute" in WC.GEOMETRY_CRITICAL
    assert "zoom_absolute" in WC.GEOMETRY_CRITICAL
    assert "exposure_time_absolute" not in WC.GEOMETRY_CRITICAL
    assert "white_balance_temperature" not in WC.GEOMETRY_CRITICAL


def _png_bytes(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height)


def test_png_size_reads_the_header(tmp_path):
    """The resolution check must not need an image library on the capture host."""
    p = tmp_path / "f.png"
    p.write_bytes(_png_bytes(1920, 1080))
    assert WC.png_size(str(p)) == (1920, 1080)


def test_png_size_rejects_a_non_png(tmp_path):
    p = tmp_path / "f.png"
    p.write_bytes(b"not a png at all, definitely not" + b"\x00" * 32)
    assert WC.png_size(str(p)) is None


def test_pixel_format_map_prefers_uncompressed():
    """YUYV is the default because MJPEG artefacts land on the very edges the fit detects."""
    assert WC.PIXFMT["YUYV"] == "yuyv422"
    assert WC.PIXFMT["MJPG"] == "mjpeg"


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
