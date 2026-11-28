"""Export the speaker notes to a printable Word file, with every {fact} filled from results/tables.

Uses only the standard library to write the .docx, so it runs wherever the deck builds.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_pptx  # noqa: E402

OUT = ROOT / "presentation" / "speaker_notes.docx"
CHARS_PER_MIN = 235

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def _para(text: str, *, size: int = 22, bold: bool = False, color: str = "17171A", before: int = 60,
          after: int = 120) -> str:
    return (
        f'<w:p><w:pPr><w:spacing w:before="{before}" w:after="{after}" w:line="340" w:lineRule="auto"/></w:pPr>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/>'
        f'{"<w:b/>" if bold else ""}<w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def main() -> None:
    raw = build_pptx.NOTES_MD.read_text(encoding="utf-8")
    titles = dict(re.findall(r"^## \[([A-Za-z0-9_]+)\]\s*([^\n]*)$", raw, flags=re.MULTILINE))
    notes = build_pptx.parse_speaker_notes(raw)
    facts = build_pptx.Facts(build_pptx.compute_facts(build_pptx.load_tables(build_pptx.TABLES)))
    fmt = build_pptx.NotesFormatter(facts)
    blocks, main_chars = [], 0
    for page, (sid, body) in enumerate(notes, start=1):
        text = fmt.format(body)
        n_chars = len(re.sub(r"\s+", "", text))
        label = f"第 {page} 页" if sid.startswith("S") else sid[:3]
        if sid.startswith("S"):
            main_chars += n_chars
        blocks.append(_para(f"{label}　{titles.get(sid, '')}（约 {round(n_chars / CHARS_PER_MIN * 60)} 秒）",
                            size=26, bold=True, before=260, after=80))
        for line in text.split("\n\n"):
            if line.strip():
                blocks.append(_para(line.strip()))
    n_main = sum(1 for sid, _ in notes if sid.startswith("S"))
    head = [
        _para("深度聚类：范式演进、对照实验与方法选择 — 讲稿", size=34, bold=True, before=0, after=120),
        _para(f"第九组 · 正文 {n_main} 页约 {main_chars} 字，按每分钟 {CHARS_PER_MIN} 字约 {main_chars / CHARS_PER_MIN:.0f} 分钟；"
              "附录 A01–A06 用于答疑。数字与幻灯片同源，均来自 results/tables。", size=20, color="6B6B73", before=0,
              after=240),
    ]
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
           + "".join(head + blocks)
           + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
             'w:left="1134"/></w:sectPr></w:body></w:document>')
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("word/document.xml", doc.encode("utf-8"))
    print(f"Exported {len(notes)} slide notes ({main_chars} chars in the main talk) -> {OUT}")


if __name__ == "__main__":
    main()
