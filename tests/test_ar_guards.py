"""Soundness guards for the AR marker endpoints."""
import numpy as np
import pytest

pytest.importorskip("cv2")
from fastapi import HTTPException

from app.routers import ar


def test_resolution_guard_rejects_mismatch():
    img = np.zeros((100, 200, 3), np.uint8)   # w=200, h=100

    # Matching resolution → no raise.
    ar._require_resolution_match({"image_size": [200, 100]}, img)
    # No image_size in profile → can't check → no raise.
    ar._require_resolution_match({}, img)
    # Mismatch → reject (would otherwise give a silently-wrong pose).
    with pytest.raises(HTTPException):
        ar._require_resolution_match({"image_size": [6048, 6048]}, img)


def test_run_id_of():
    assert ar._run_id_of("feac0770_150x75x18 1790lg.step") == "feac0770"
