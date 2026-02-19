"""QR code rendering for terminal display."""

from __future__ import annotations

import sys


def render_qr_terminal(url: str, *, border: int = 1) -> str:
    """Render a QR code as ASCII art for terminal display.

    Uses unicode block characters for compact rendering.
    Falls back to a simple text URL if qrcode lib not available.
    """
    try:
        import qrcode  # type: ignore[import-untyped]
    except ImportError:
        return f"[QR code unavailable — install qrcode: pip install qrcode]\n{url}"

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=1,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)

    modules = qr.modules
    if not modules:
        return url

    lines: list[str] = []
    row_count = len(modules)

    # Process two rows at a time using half-block characters.
    # Top row dark + bottom row dark = full block
    # Top row dark + bottom row light = upper half block
    # Top row light + bottom row dark = lower half block
    # Top row light + bottom row light = space
    for y in range(0, row_count, 2):
        row_top = modules[y]
        row_bot = modules[y + 1] if y + 1 < row_count else [False] * len(row_top)
        line_chars: list[str] = []
        for x in range(len(row_top)):
            top = row_top[x]
            bot = row_bot[x]
            if top and bot:
                line_chars.append("\u2588")  # full block
            elif top and not bot:
                line_chars.append("\u2580")  # upper half block
            elif not top and bot:
                line_chars.append("\u2584")  # lower half block
            else:
                line_chars.append(" ")
        lines.append("".join(line_chars))

    return "\n".join(lines)


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
