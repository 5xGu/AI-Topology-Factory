'''
Expose markdown extraction tools to models.
'''

from langchain_core.tools import tool
from agentic_dd.extraction.markdownExtraction import MarkdownDocument, Section

_doc_cache: dict[str, MarkdownDocument] = {}

def _get_doc(path: str) -> MarkdownDocument:
    if path not in _doc_cache:
        with open(path, encoding="utf-8") as f:
            _doc_cache[path] = MarkdownDocument(f.read())
    return _doc_cache[path]

@tool
def md_outline(path: str) -> list[dict]:
    """ Get the section outline of a markdown file:
        - section id
        - section level
        - section titel
        - sections structural context (breadcrumbs)
        - section length as number of characters
    """
    return _get_doc(path).outline_json()

@tool
def md_get_section(path: str, section_id: int, include_subsections: bool = False, max_chars: int | None = None) -> dict:
    """ Get the content of one markdown section by id, optionally including subsections. Use md_outline first to get the section id of an intersting section."""
    return _get_doc(path).get_section_json(section_id, include_subsections=include_subsections, max_chars=max_chars)

@tool
def md_search(path: str, query: str, regex: bool = False) -> list[dict]:
    """ Search a markdown document for a literal string or regex pattern. """
    return _get_doc(path).search_json(query, regex=regex)

@tool
def md_tables(path: str) -> list[dict]:
    """ List all tables found in a markdown document (defined by the pipe symbol). """
    return _get_doc(path).tables_json()

@tool
def md_images(path: str) -> list[dict]:
    """ List all image references (with local context) in a markdown document. """
    return _get_doc(path).images_json()


# -------- Register tools -------- #
from agentic_dd.agentlib.tools import register_tool

register_tool('md_outline', md_outline)
register_tool('md_get_section', md_get_section)
register_tool('md_search', md_search)
register_tool('md_tables', md_tables)
register_tool('md_images', md_images)