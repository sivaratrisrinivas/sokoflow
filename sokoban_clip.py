"""Tiny paletted GIF encoder for a denoise clip. No extra dependencies."""

from __future__ import annotations

import struct

import numpy as np

from sokoban_render import PIECE_COLORS

# Palette indices match raster_board below. GIF needs a power-of-two table.
PALETTE_RGB = [
    PIECE_COLORS["floor"],
    PIECE_COLORS["wall"],
    PIECE_COLORS["box"],
    PIECE_COLORS["goal"],
    PIECE_COLORS["player"],
    PIECE_COLORS["box_on_goal"],
    "#F3EEE4",  # cream chrome
    "#D7E6F0",  # sky trail
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def raster_board(grid, targets, trail=None, cell: int = 10, gap: int = 1, pad: int = 4) -> np.ndarray:
    """8×8 board as palette indices (uint8)."""
    trail_set = set(tuple(p) for p in (trail or []))
    inner = 8 * cell + 7 * gap
    width = inner + 2 * pad
    img = np.full((width, width), 6, dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            y0 = pad + r * (cell + gap)
            x0 = pad + c * (cell + gap)
            v = int(grid[r][c])
            on_goal = bool(targets[r][c]) if targets is not None else False
            img[y0 : y0 + cell, x0 : x0 + cell] = 0
            inset = max(2, cell // 5)
            if v == 1:
                img[y0 : y0 + cell, x0 : x0 + cell] = 1
            elif v == 3:
                img[y0 + inset : y0 + cell - inset, x0 + inset : x0 + cell - inset] = 2
            elif v == 5:
                img[y0 + inset : y0 + cell - inset, x0 + inset : x0 + cell - inset] = 5
            elif v == 4 or (v == 0 and on_goal):
                cy, cx = y0 + cell // 2, x0 + cell // 2
                img[cy - 1 : cy + 2, cx - 1 : cx + 2] = 3
            elif v == 2:
                if on_goal:
                    cy, cx = y0 + cell // 2, x0 + cell // 2
                    img[cy - 1 : cy + 2, cx - 1 : cx + 2] = 3
                img[y0 + inset : y0 + cell - inset, x0 + inset : x0 + cell - inset] = 4
            if (r, c) in trail_set and v in (0, 4):
                cy, cx = y0 + cell // 2, x0 + cell // 2
                img[max(y0, cy - 1) : cy + 1, max(x0, cx - 1) : cx + 1] = 7
    return img


def _lzw_uncompressed(indices: np.ndarray) -> bytes:
    """GIF image data with min-code-size 8 (uncompressed 9-bit codes)."""
    clear, eoi = 256, 257
    acc = 0
    nbits = 0
    stream = bytearray()

    def emit(code: int) -> None:
        nonlocal acc, nbits
        acc |= code << nbits
        nbits += 9
        while nbits >= 8:
            stream.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8

    emit(clear)
    count = 0
    for pix in indices.reshape(-1):
        emit(int(pix) & 0xFF)
        count += 1
        if count == 126:
            emit(clear)
            count = 0
    emit(eoi)
    if nbits:
        stream.append(acc & 0xFF)

    data = bytearray([8])
    buf = bytes(stream)
    for i in range(0, len(buf), 255):
        chunk = buf[i : i + 255]
        data.append(len(chunk))
        data.extend(chunk)
    data.append(0)
    return bytes(data)


def denoise_gif_bytes(frames: list[dict], targets, delay_cs: int = 14) -> bytes:
    """Animated GIF of denoise theater boards. Cheap, no extra packages."""
    if not frames:
        raise ValueError("no denoise frames")
    rasters = [raster_board(f["grid"], targets, f.get("trail")) for f in frames]
    height, width = rasters[0].shape
    palette = bytearray()
    for hex_color in PALETTE_RGB:
        palette.extend(_hex_to_rgb(hex_color))
    palette.extend(b"\x00" * (3 * (256 - len(PALETTE_RGB))))

    out = bytearray()
    out.extend(b"GIF89a")
    out.extend(struct.pack("<HH", width, height))
    out.append(0xF7)  # GCT flag, 8-bit
    out.append(0)  # bg
    out.append(0)  # aspect
    out.extend(palette)
    # Netscape loop
    out.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")

    for raster in rasters:
        out.extend(b"\x21\xf9\x04\x04")
        out.extend(struct.pack("<H", delay_cs))
        out.append(0)
        out.append(0)
        out.extend(b"\x2c")
        out.extend(struct.pack("<HHHH", 0, 0, width, height))
        out.append(0)
        out.extend(_lzw_uncompressed(raster))
    out.append(0x3B)
    return bytes(out)


def denoise_gif_data_uri(frames: list[dict], targets, delay_cs: int = 14) -> str | None:
    try:
        import base64

        raw = denoise_gif_bytes(frames, targets, delay_cs=delay_cs)
        if len(raw) > 400_000:
            return None
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/gif;base64,{b64}"
    except Exception:
        return None
