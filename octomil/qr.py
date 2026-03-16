"""QR code rendering for terminal display and file export."""

from __future__ import annotations

import io
import sys
import urllib.parse


def build_deep_link(token: str, host: str) -> str:
    """Build an ``octomil://`` deep link URL for QR code pairing.

    Uses custom scheme ``octomil://pair?token=CODE`` which is handled by
    the Android and iOS companion apps without requiring domain verification.
    A non-default host is appended as ``&host=`` so the app knows which server.
    """
    default_host = "https://api.octomil.com/api/v1"
    params = {"token": token}
    if host != default_host:
        params["host"] = host
    return f"octomil://pair?{urllib.parse.urlencode(params)}"


def build_custom_scheme_link(token: str, host: str) -> str:
    """Build an ``octomil://`` deep link for manual opening."""
    default_host = "https://api.octomil.com/api/v1"
    params = {"token": token}
    if host != default_host:
        params["host"] = host
    return f"octomil://pair?{urllib.parse.urlencode(params)}"


def render_qr_terminal(url: str, *, border: int = 4) -> str:
    """Render a QR code as terminal text using segno's built-in renderer.

    Uses ``segno.make_qr`` to guarantee a full QR Code (not Micro QR).
    Error correction M (15% redundancy) keeps module count low.
    ``border=4`` is the QR spec minimum quiet zone.

    Falls back to a plain text URL if segno is not installed.
    """
    try:
        import segno
    except ImportError:
        return f"[QR code unavailable — install segno: pip install segno]\n{url}"

    qr = segno.make_qr(url, error="m")
    buf = io.StringIO()
    qr.terminal(out=buf, border=border, compact=True)
    return buf.getvalue().rstrip("\n")


def save_qr_svg(url: str, path: str, *, border: int = 4, scale: int = 4) -> None:
    """Save QR code as an SVG file for reliable scanning."""
    import segno

    qr = segno.make_qr(url, error="m")
    qr.save(path, scale=scale, border=border)


def print_qr_code(url: str, *, label: str = "") -> str:
    """Print a QR code to stdout with optional label below it.

    Returns the rendered QR string for testability.
    """
    qr_text = render_qr_terminal(url)
    output = qr_text
    if label:
        output += f"\n{label}"
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    return qr_text
