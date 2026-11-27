"""Edit the deck through the running PowerPoint process."""

from __future__ import annotations

import sys
from pathlib import Path

import win32com.client

PPTX = Path(__file__).resolve().parents[1] / "presentation" / "deep-clustering-evolution-selection-guide.pptx"


def application():
    try:
        return win32com.client.GetActiveObject("PowerPoint.Application")
    except Exception:
        app = win32com.client.Dispatch("PowerPoint.Application")
        app.Visible = True
        return app


def open_deck(path: Path = PPTX):
    app = application()
    app.Visible = True
    target = str(path.resolve())
    for presentation in app.Presentations:
        try:
            if str(Path(presentation.FullName).resolve()).lower() == target.lower():
                return app, presentation
        except Exception:
            continue
    return app, app.Presentations.Open(target, WithWindow=True)


def export_slides(indexes: list[int], directory: Path) -> None:
    _app, presentation = open_deck()
    directory.mkdir(parents=True, exist_ok=True)
    for index in indexes:
        presentation.Slides(index).Export(str(directory / f"slide-{index:02d}.png"), "PNG", 1920, 1080)


if __name__ == "__main__":
    indexes = [int(value) for value in sys.argv[1:]] or [1, 6, 9, 15]
    export_slides(indexes, PPTX.parent / "preview")
    print("opened", PPTX)
