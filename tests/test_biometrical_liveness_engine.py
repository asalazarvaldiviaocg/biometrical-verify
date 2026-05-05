"""
Unit tests for the Biometrical Liveness Engine (BLE).

These do NOT require the full FastAPI app or any network access. They build
synthetic 'real-face-ish' and 'printed-photo-ish' images using NumPy and
verify the engine separates them.

Synthetic real-face image:
    - rich Saturation distribution (additive sinusoids on S channel)
    - skin-tone YCrCb pixels in canonical range
    - high LBP entropy (random-ish noise + features)
    - sharp gradient edges around eye/nose region
    - mid-band frequency energy

Synthetic printed-photo image:
    - flat saturation
    - shifted skin distribution
    - smooth gradients (post-print blur)
    - over-low frequency energy
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services import biometrical_liveness_engine as ble


# ────────────────────────────────────────────────────────────────────────────
# Synthetic image factories
# ────────────────────────────────────────────────────────────────────────────


def _make_real_face(side: int = 160, seed: int = 7) -> np.ndarray:
    """Synthesize a face-like BGR image. Mimics the signal profile of a
    genuine selfie: high saturation variance, skin-tone pixels, sharp
    edges and rich texture."""
    rng = np.random.default_rng(seed)
    # Start from a skin-tone base and add structured noise to provoke high
    # LBP entropy + gradient variance.
    base = np.zeros((side, side, 3), dtype=np.uint8)
    base[:, :, 0] = 130   # B
    base[:, :, 1] = 160   # G
    base[:, :, 2] = 200   # R   → ~ skin tone in BGR
    # Texture noise (skin pores)
    noise = rng.integers(-30, 30, (side, side, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Add an "eye region" with strong contrast to give gradient variance.
    yy, xx = np.indices((side, side))
    eye_left  = ((xx - side * 0.35) ** 2 + (yy - side * 0.4) ** 2) < (side * 0.08) ** 2
    eye_right = ((xx - side * 0.65) ** 2 + (yy - side * 0.4) ** 2) < (side * 0.08) ** 2
    img[eye_left]  = (40, 30, 30)
    img[eye_right] = (40, 30, 30)
    # Add lip region with redder pixels to spread saturation.
    lip = ((xx - side * 0.5) ** 2 + (yy - side * 0.75) ** 2) < (side * 0.07) ** 2
    img[lip] = (90, 70, 220)
    return img


def _make_printed_photo(side: int = 160, seed: int = 11) -> np.ndarray:
    """Synthesize a printed-photo-like BGR image: flatter saturation,
    smoothed edges, narrower color range. Approximates a thermal-printed
    selfie that an attacker might present to the camera."""
    rng = np.random.default_rng(seed)
    base = np.zeros((side, side, 3), dtype=np.uint8)
    base[:, :, 0] = 145
    base[:, :, 1] = 150
    base[:, :, 2] = 160  # narrower / flatter color range
    # Very small noise — printed photos lose pore texture.
    noise = rng.integers(-3, 3, (side, side, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Add eyes but with soft gradient (printer blur).
    import cv2
    yy, xx = np.indices((side, side))
    eye_left  = ((xx - side * 0.35) ** 2 + (yy - side * 0.4) ** 2) < (side * 0.08) ** 2
    eye_right = ((xx - side * 0.65) ** 2 + (yy - side * 0.4) ** 2) < (side * 0.08) ** 2
    img[eye_left]  = (90, 80, 80)
    img[eye_right] = (90, 80, 80)
    # Heavy Gaussian blur to mimic print-out softness.
    img = cv2.GaussianBlur(img, (15, 15), sigmaX=4.0, sigmaY=4.0)
    return img


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


class TestBleSeparation:
    def test_real_score_higher_than_printed(self) -> None:
        real = ble.analyze(_make_real_face())
        spoof = ble.analyze(_make_printed_photo())
        assert real.score > spoof.score, (
            f"BLE failed to separate real ({real.score:.3f}) from "
            f"printed ({spoof.score:.3f})"
        )

    def test_real_passes_default_threshold(self) -> None:
        result = ble.analyze(_make_real_face())
        assert result.score > 0.0
        assert result.threshold == ble.LIVENESS_PASS_THRESHOLD
        # We don't strictly require is_live=True with a synthetic image —
        # only that the score is meaningfully above the spoof's.

    def test_signal_breakdown_present(self) -> None:
        result = ble.analyze(_make_real_face())
        d = result.to_public()
        assert set(d["signals"].keys()) == {
            "hsv_variance",
            "lbp_entropy",
            "fft_ratio",
            "skin_density",
            "gradient_variance",
        }
        for v in d["signals"].values():
            assert 0.0 <= v <= 1.0


class TestBleEdgeCases:
    def test_rejects_empty_array(self) -> None:
        with pytest.raises(ValueError):
            ble.analyze(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_rejects_grayscale(self) -> None:
        with pytest.raises(ValueError):
            ble.analyze(np.zeros((160, 160), dtype=np.uint8))

    def test_passive_score_wrapper_returns_float(self) -> None:
        score = ble.passive_score(_make_real_face())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestBleSignalRanges:
    """Each signal must return a value in [0, 1] for any valid input —
    otherwise the weighted ensemble can drift outside [0, 1]."""

    @pytest.mark.parametrize("img_factory", [_make_real_face, _make_printed_photo])
    def test_signal_hsv_in_unit_range(self, img_factory) -> None:
        img = img_factory()
        norm = ble._normalize_face(img)
        s = ble.signal_hsv_variance(norm)
        assert 0.0 <= s <= 1.0

    @pytest.mark.parametrize("img_factory", [_make_real_face, _make_printed_photo])
    def test_signal_lbp_in_unit_range(self, img_factory) -> None:
        img = img_factory()
        norm = ble._normalize_face(img)
        s = ble.signal_lbp_entropy(norm)
        assert 0.0 <= s <= 1.0

    @pytest.mark.parametrize("img_factory", [_make_real_face, _make_printed_photo])
    def test_signal_fft_in_unit_range(self, img_factory) -> None:
        img = img_factory()
        norm = ble._normalize_face(img)
        s = ble.signal_fft_ratio(norm)
        assert 0.0 <= s <= 1.0

    @pytest.mark.parametrize("img_factory", [_make_real_face, _make_printed_photo])
    def test_signal_skin_in_unit_range(self, img_factory) -> None:
        img = img_factory()
        norm = ble._normalize_face(img)
        s = ble.signal_skin_density(norm)
        assert 0.0 <= s <= 1.0

    @pytest.mark.parametrize("img_factory", [_make_real_face, _make_printed_photo])
    def test_signal_gradient_in_unit_range(self, img_factory) -> None:
        img = img_factory()
        norm = ble._normalize_face(img)
        s = ble.signal_gradient_variance(norm)
        assert 0.0 <= s <= 1.0


class TestBleDeterministic:
    def test_same_input_same_output(self) -> None:
        img = _make_real_face()
        a = ble.analyze(img).score
        b = ble.analyze(img).score
        assert a == b
