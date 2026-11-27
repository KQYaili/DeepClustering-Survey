"""Export speaker notes from presentation/speaker_notes.md to a clean printable Word .docx.

Uses pure Python stdlib (zipfile + xml) so it runs reliably without extra dependencies.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "presentation" / "speaker_notes.md"
OUT = ROOT / "presentation" / "speaker_notes.docx"

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


def _para(text: str, *, size_half_pt: int = 22, bold: bool = False, color: str = "111722",
          space_before: int = 60, space_after: int = 120) -> str:
    b_tag = "<w:b/>" if bold else ""
    safe = escape(text)
    return (
        f'<w:p><w:pPr><w:spacing w:before="{space_before}" w:after="{space_after}" w:line="320" w:lineRule="auto"/></w:pPr>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/>'
        f'{b_tag}<w:color w:val="{color}"/><w:sz w:val="{size_half_pt}"/><w:szCs w:val="{size_half_pt}"/></w:rPr>'
        f'<w:t xml:space="preserve">{safe}</w:t></w:r></w:p>'
    )


def main() -> None:
    raw = MD.read_text(encoding="utf-8")
    sections = list(re.finditer(r"^## (?:\[[^\]]+\]\s+)?(第 \d+ 页[^\n]*)\n\n(.*?)(?=^## |\Z)", raw, re.MULTILINE | re.DOTALL))
    main_chars = 0
    blocks = [
        _para("《深度聚类：范式演进与方法选型综述》逐页汇报讲稿", size_half_pt=34, bold=True, color="073DA5", space_before=0, space_after=140),
        _para("第九组课程汇报 · 24 页正文（约 23–25 分钟）+ 6 页技术附录（A1–A6 答疑备查） · 与 PPT 备注完全同源", size_half_pt=20, color="475569", space_before=0, space_after=220),
    ]
    for idx, m in enumerate(sections, start=1):
        heading = m.group(1).strip()
        body = m.group(2).strip()
        n_chars = len(re.sub(r"\s+", "", body))
        est_sec = round(n_chars / 235 * 60)
        if idx <= 24:
            main_chars += n_chars
        blocks.append(
            _para(f"{heading}（约 {n_chars} 字 · 预计 {est_sec} 秒）", size_half_pt=26, bold=True, color="073DA5", space_before=220, space_after=80)
        )
        for line in body.split("\n\n"):
            line = line.strip()
            if line:
                blocks.append(_para(line, size_half_pt=22, color="111722", space_before=40, space_after=100))

    summary_min = main_chars / 235.0
    blocks.insert(
        2,
        _para(f"正文 24 页合计约 {main_chars} 字（按 235 字/分估算约 {summary_min:.1f} 分钟，预留约 5 分钟答疑）。", size_half_pt=20, bold=True, color="0F766E", space_before=0, space_after=200),
    )

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(blocks)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr>'
        "</w:body></w:document>"
    )

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
    print(f"Exported {len(sections)} slide notes ({main_chars} chars, ~{summary_min:.1f} min) -> {OUT}")


if __name__ == "__main__":
    main()
