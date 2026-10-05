"""
Lightweight regex based syntactical elements markdown extractor, mostly for GWDG docling returned markdown.
Because most of the .md files we have are returned as parses of .pdfs, they often share the types
of structuring components, which are mostly restricted to headings, tables and images.

Exceptions to this mostly do not use other structuring components, but rather none at all.
These are typically also comparatively short documents.

Assumptions about input:
  - Headings use ATX style: '#', '##', ... '######'
  - Tables are GFM-style pipe tables (header row + separator row of dashes)
  - Images are replaced by tokens like "picture-1.png", "picture-2.png", ...
  - Fenced code blocks (```...```) are respected and excluded from
    heading/table detection to avoid false positives.

Design: a single MarkdownDocument class parses the text once (sections,
tables) and exposes query methods for images, arbitrary string search,
and context extraction. All parsing is regex/line based.

"""

from __future__ import annotations

import re
import bisect
from dataclasses import asdict, dataclass, field
from typing import List, Optional


######### Predefined regex patterns #########

_HEADING_RE = re.compile(r'^(#{1,6})[ \t]+(.*?)[ \t]*$', re.MULTILINE)
_IMAGE_RE = re.compile(r'\bpicture-\d+\b', re.IGNORECASE)
_FENCE_RE = re.compile(r'^\s*```')
_CELL_SEP_RE = re.compile(r'^:?-{2,}:?$')


def _is_table_row(line: str) -> bool:
    line = line.strip()
    return bool(line) and '|' in line


def _is_separator_row(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    cells = [c.strip() for c in line.strip('|').split('|')]
    cells = [c for c in cells if c != '']
    return bool(cells) and all(_CELL_SEP_RE.match(c) for c in cells)


######### Data Classes #########
@dataclass
class Section:
    id: int
    level: int
    title: str
    start: int              # char offset of heading line
    content_start: int      # char offset right after heading line
    end: int                # end of section direct subordinate content (until next heading, any level)
    subtree_end: int        # end including subsections (next heading level<=self.level)
    breadcrumb: List[str]   # Chain of headings from document root to current heading, for disambiguation if necessary
    _source: str = field(repr=False, default="")

    @property
    def heading_line(self) -> str:
        return "#" * self.level + " " + self.title

    @property
    def content(self) -> str:
        """Text belonging only to this heading (excludes subsections)."""
        return self._source[self.content_start:self.end]

    @property
    def full_content(self) -> str:
        """Text belonging to this heading and all its subsections."""
        return self._source[self.content_start:self.subtree_end]


@dataclass
class Table:
    id: int
    start: int
    end: int
    raw: str
    rows: List[List[str]]
    section_title: Optional[str]


@dataclass
class ImageReference:
    id: int
    tag: str
    position: int
    context: str
    section_title: Optional[str]


@dataclass
class SearchResult:
    match: str
    position: int
    context: str
    section: Optional[Section]


class MarkdownDocument:
    ''' Lightweight regex parser for Markdown files. Uses ids and dataclasses to represent results in .jsons for agentic operations. '''
    def __init__(self, text: str):
        self.text = text
        self._line_starts = self._compute_line_starts()
        self._fenced_lines = self._compute_fenced_lines()

        self.sections: List[Section] = self._parse_sections()
        self.tables: List[Table] = self._parse_tables()


    ######### General util methods #########

    def _compute_line_starts(self) -> List[int]:
        starts = [0]
        for i, ch in enumerate(self.text):
            if ch == "\n":
                starts.append(i + 1)
        return starts

    def _line_of_offset(self, offset: int) -> int:
        return bisect.bisect_right(self._line_starts, offset) - 1

    def _compute_fenced_lines(self) -> set:
        fenced = set()
        in_fence = False
        for idx, line in enumerate(self.text.split("\n")):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                fenced.add(idx)
        return fenced

    def _extract_context(self, start: int, end: int, window: int = 200) -> str:
        left = max(0, start - window)
        right = min(len(self.text), end + window)
        return self.text[left:right].strip()


    ######### Section extraction #########

    def _parse_sections(self) -> List[Section]:
        raw_headings = []
        for m in _HEADING_RE.finditer(self.text):
            if self._line_of_offset(m.start()) in self._fenced_lines:
                continue
            raw_headings.append((m.start(), len(m.group(1)), m.group(2).strip(), m.end()))

        if not raw_headings:
            # whole document is one anonymous section
            return [Section(
                id=0, level=0, title="", start=0, content_start=0,
                end=len(self.text), subtree_end=len(self.text),
                breadcrumb=[], _source=self.text,
            )]
        sections: List[Section] = []
        stack: List[tuple] = []  # (level, title)

        for i, (start, level, title, content_start) in enumerate(raw_headings):
            end = raw_headings[i + 1][0] if i + 1 < len(raw_headings) else len(self.text)

            # subtree_end: next heading with level <= this one, or EOF
            subtree_end = len(self.text)
            for j in range(i + 1, len(raw_headings)):
                if raw_headings[j][1] <= level:
                    subtree_end = raw_headings[j][0]
                    break

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            breadcrumb = [t for _, t in stack]

            sections.append(Section(i, level, title, start, content_start, end,
                                     subtree_end, breadcrumb, self.text))
        return sections

    def get_section_for_position(self, pos: int) -> Optional[Section]:
        """Return the most specific (deepest) section containing `pos`."""
        for sec in self.sections:
            if sec.start <= pos < sec.end:
                return sec
        # fallback: pos might fall in a parent's own_content gap due to
        # end being "own content only" boundaries — check subtree ranges
        for sec in reversed(self.sections):
            if sec.start <= pos < sec.subtree_end:
                return sec
        return None

    def get_section_by_title(self, title: str, case_sensitive: bool = False) -> Optional[Section]:
        for sec in self.sections:
            if case_sensitive:
                if sec.title == title:
                    return sec
            elif sec.title.lower() == title.lower():
                return sec
        return None

    def outline(self) -> str:
        return "\n".join(f"{'  ' * (s.level - 1)}- {s.title or '(untitled)'}"
                          for s in self.sections)

    ######### Table extraction #########

    def _parse_tables(self) -> List[Table]:
        lines = self.text.split("\n")
        offsets = []
        cum = 0
        for line in self.text.splitlines(keepends=True):
            offsets.append(cum)
            cum += len(line)

        tables: List[Table] = []
        table_id = 0 
        i, n = 0, len(lines)
        while i < n:
            if (i not in self._fenced_lines and i + 1 < n
                    and _is_table_row(lines[i])
                    and _is_separator_row(lines[i + 1])):
                start_idx = i
                j = i + 2
                while j < n and j not in self._fenced_lines and _is_table_row(lines[j]):
                    j += 1
                block = lines[start_idx:j]
                start = offsets[start_idx]
                end = offsets[j - 1] + len(lines[j - 1]) if j - 1 < len(offsets) else len(self.text)
                raw = "\n".join(block)
                rows = self._parse_table_rows(block)
                sec = self.get_section_for_position(start)
                tables.append(Table(table_id, start, end, raw, rows,
                                     sec.title if sec else None))
                i = j
            else:
                i += 1
        return tables

    @staticmethod
    def _parse_table_rows(block_lines: List[str]) -> List[List[str]]:
        rows = []
        for idx, line in enumerate(block_lines):
            if idx == 1:
                continue  # skip separator row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cells)
        return rows

    ######### Image extraction #########

    def images(self, context_chars: int = 200) -> List[ImageReference]:
        ''' Returns images and local context '''
        results = []
        for img_id, m in enumerate(_IMAGE_RE.finditer(self.text)):
            if self._line_of_offset(m.start()) in self._fenced_lines:
                continue
            pos = m.start()
            ctx = self._extract_context(pos, m.end(), window=context_chars)
            sec = self.get_section_for_position(pos)
            results.append(ImageReference(img_id, m.group(0), pos, ctx,
                                           sec.title if sec else None))
        return results

    ######### Search #########

    def search(self, query: str, *, case_sensitive: bool = False,
               regex: bool = False, context_chars: int = 200) -> List[SearchResult]:
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = query if regex else re.escape(query)
        results = []
        for m in re.finditer(pattern, self.text, flags):
            pos = m.start()
            sec = self.get_section_for_position(pos)
            ctx = self._extract_context(pos, m.end(), window=context_chars)
            results.append(SearchResult(m.group(0), pos, ctx, sec))
        return results

    def get_context(self, position: int, window: int = 200) -> str:
        return self._extract_context(position, position, window=window)


    ######### Methods for Agent support #########

    def outline_json(self) -> list[dict]:
        """ Lightweight index for agentic file navigation """
        return [
            {"id": s.id, "level": s.level, "title": s.title,
             "breadcrumb": s.breadcrumb, "char_len": s.end - s.content_start}
            for s in self.sections
        ]

    def get_section_json(self, section_id: int, *, include_subsections=False, max_chars: int = None) -> dict:
        sec = self.sections[section_id]
        content = sec.full_content if include_subsections else sec.content
        truncated = False
        if max_chars:
            truncated = len(content) > max_chars
            content = content[:max_chars]
        return {
            "id": sec.id, "title": sec.title, "breadcrumb": sec.breadcrumb,
            "content": content, "truncated": truncated,
        }

    def search_json(self, query: str, **kwargs) -> list[dict]:
        return [
            {"match": r.match, "position": r.position, "context": r.context,
             "section_id": r.section.id if r.section else None,
             "section_title": r.section.title if r.section else None}
            for r in self.search(query, **kwargs)
        ]

    def tables_json(self) -> list[dict]:
        return [asdict(t) for t in self.tables]

    def images_json(self) -> list[dict]:
        return [asdict(i) for i in self.images()]





#Test
'''
if __name__ == "__main__":
    from jsonExtraction import *
    fp = '/home/tp/Desktop/BA/parsedData/Corruption_in_Ukraine._Perceptions_and_Experience_-_Waves_1-8_2007-2024Data_from_eight_waves_of_nationwide_representative_surveys_on_corruption_by_KIIS_with_USAID-funded_projects_MSI_ACTION_2007_2009_Pacts_UNITER_2011_2015_ENGAGE_2018_-2024/metadata/DiscussData-Corruption_in_Ukraine_2007-2024-Corr2015_Ank_RU.json'
    jf = load_json(fp)
    schema = list(s for s in schema(jf) if s.lvl == 0) 
    md_string = get_by_path(jf, schema[2].path)

    doc = MarkdownDocument(md_string)
    #for s in doc.sections:
     #   print(len(s.breadcrumb))
    #print(doc.tables)
    #print(doc.images())
    #print(doc.search("picture-3"))
'''