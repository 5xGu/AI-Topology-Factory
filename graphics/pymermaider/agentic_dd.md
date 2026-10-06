classDiagram
    class ChunkPlan {
        - \_\_init__(self, needs_chunking, num_chunks, chunk_size_chars) None
    }

    class ConditionSpec {
        + dict model_config
        + str name
        + str type
        + Dict[str, Any] params
        + Optional[List[ReadSpec]] reads
    }

    class EdgeSpec {
        + dict model_config
        + str source
        + Optional[str] target
        + Optional[str] condition
        + Optional[List[str]] condition_targets
        - \_check_shape(self) "EdgeSpec"
    }

    class FileTokenCounts {
        + Dict[str, int] filename
    }

    class GraphState {
        + List[str] input
        + Optional[str] task
        + Annotated[List[Dict[str, Any]], operator.add] results
        + Optional[int] chunk_index
    }

    class ImageReference {
        <<dataclass>>
        + int id
        + str tag
        + int position
        + str context
        + Optional[str] section_title
    }

    class MetadataResponse {
        + Optional[str] Title
        + Optional[str] Subtitle
        + Optional[str] Creators
        + Optional[Literal["Person", "Institution"]] Creator_type
        + Optional[str] Institutional_affiliation
        + Optional[Literal[CATEGORIES]] Main_Category
        + Optional[List[Literal[CATEGORIES]]] Additional_Categories
        + Optional[str] Publication_date
        + Optional[str] Date_period_of_data_creation_from
        + Optional[str] Date_period_of_data_creation_to
        + Optional[str] Date_of_data_creation_text
        + Optional[str] Time_period_covered_in_data_collection_from
        + Optional[str] Time_period_covered_in_data_collection_to
        + Optional[str] Time_period_covered_text
        + Optional[List[str]] Sources_of_data
        + Optional[List[Literal["archival documents", "audio", "images", "interviews", "statistics", "survey", "text document", "video", "other"]]] Data_types
        + Optional[str] Datatype_text
        + Optional[List[Literal["Armenia", "Azerbaijan", "Belarus", "Estonia", "GDR", "Georgia", "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Moldova", "Other", "Russia", "Tajikistan", "Turkmenistan", "Ukraine", "USSR", "Uzbekistan"]]] Countries
        + Optional[List[str]] Languages
        + Optional[List[str]] Disciplines
        + Optional[List[str]] Keywords
        + Optional[str] Related_dataset
        + Optional[str] Related_dataset_text
        + Optional[str] Related_publications
        + Optional[List[str]] Related_projects
        + Optional[str] Funding
        + Optional[List[str]] Methods_of_data_collection
        + Optional[List[str]] Methods_of_data_analysis
    }

    class IncrementalSynthesis {
        + str updated_summary
        + MetadataResponse updated_metadata
        + str notes
    }

    class MarkdownDocument {
        - \_\_init__(self, text) None
        - \_compute_line_starts(self) List[int]
        - \_line_of_offset(self, offset) int
        - \_compute_fenced_lines(self) set
        - \_extract_context(self, start, end, window) str
        - \_parse_sections(self) List[Section]
        + get_section_for_position(self, pos) Optional[Section]
        + get_section_by_title(self, title, case_sensitive) Optional[Section]
        + outline(self) str
        - \_parse_tables(self) List[Table]
        - @staticmethod \_parse_table_rows(block_lines) List[List[str]]$
        + images(self, context_chars) List[ImageReference]
        + search(self, query, *, case_sensitive, regex, context_chars) List[SearchResult]
        + get_context(self, position, window) str
        + outline_json(self) list[dict]
        + get_section_json(self, section_id, *, include_subsections, max_chars) dict
        + search_json(self, query, **kwargs) list[dict]
        + tables_json(self) list[dict]
        + images_json(self) list[dict]
    }

    class ModelParams {
        + dict model_config
        + str provider
        + str model
        + float temperature
    }

    class ModelTypes {
        + @staticmethod has_capabilities()$
    }

    class NodeSpec {
        + dict model_config
        + str name
        + str type
        + Dict[str, Any] params
        + Optional[List[ReadSpec]] reads
        + Optional[List[str]] writes
    }

    class PlanItem {
        + str task
    }

    class PromptSpec {
        + Optional[str] template_name
        + Optional[str] system_override
        + Optional[str] instructions
        + Dict[str, Any] variables
    }

    class RateLimiter {
        - \_\_init__(self, max_calls, window_s) None
        + acquire(self)
    }

    class ReadSpec {
        + str node
        + Literal["all", "last"] mode
    }

    class ReflectionVerdict {
        + bool sufficient
        + bool conflicts_detected
        + str reasoning
    }

    class RouterDecision {
        + str route
        + str reasoning
        + float confidence
        + Optional[List[PlanItem]] plan
    }

    class SchemaEntry {
        <<dataclass>>
        + tuple[str | int, ...] path
        + tuple[str | int, ...] parent_path
        + int lvl
        + str type
    }

    class Section {
        <<dataclass>>
        + int id
        + int level
        + str title
        + int start
        + int content_start
        + int end
        + int subtree_end
        + List[str] breadcrumb
        - str \_source
        + heading_line(self) str
        + content(self) str
        + full_content(self) str
    }

    class SearchResult {
        <<dataclass>>
        + str match
        + int position
        + str context
        + Optional[Section] section
    }

    class SubagentState {
        + Annotated[List[AnyMessage], add_messages] messages
        + Any input
    }

    class SummaryEval {
        + int faithfulness_score
        + int conciseness_score
        + int coverage_score
        + str reasoning
        + str divergence_examples
    }

    class SupportedModels {
        + @staticmethod is_supported(name) bool$
        + @staticmethod get_all_supported_models() List[str]$
        + @staticmethod get_huggingface_tokenizer(model_name) str$
        + @staticmethod get_special_tokenizer_kwargs()$
    }

    class Table {
        <<dataclass>>
        + int id
        + int start
        + int end
        + str raw
        + List[List[str]] rows
        + Optional[str] section_title
    }

    class TokenCounts {
        + int model_name
    }

    class TopologySpec {
        + dict model_config
        + str entry
        + List[NodeSpec] nodes
        + List[ConditionSpec] conditions
        + List[EdgeSpec] edges
        - @classmethod \_normalize_nodes(cls, v) Any
        - @classmethod \_normalize_conditions(cls, v) Any
        - \_check_structure(self) "TopologySpec"
    }

    class ValidationStatus {
        <<enumeration>>
        + str OK
        + str FAILED
        + str ERROR
    }

    class ValidationResult {
        <<dataclass>>
        + ValidationStatus status
        + str validator
        + str message
        + valid(self) bool
    }

    class _RateLimitedWrapper {
        - \_\_init__(self, inner) None
        + invoke(self, *args, **kwargs)
        + bind_tools(self, *args, **kwargs)
        + with_structured_output(self, *args, **kwargs)
        - \_\_getattr__(self, name)
    }

    TokenCounts --|> typing.TypedDict

    FileTokenCounts --|> typing.TypedDict

    SubagentState --|> typing.TypedDict

    ModelParams --|> pydantic.BaseModel

    ReadSpec --|> pydantic.BaseModel

    PromptSpec --|> pydantic.BaseModel

    GraphState --|> typing.TypedDict

    MetadataResponse --|> pydantic.BaseModel

    PlanItem --|> pydantic.BaseModel

    RouterDecision --|> pydantic.BaseModel

    SummaryEval --|> pydantic.BaseModel

    IncrementalSynthesis --|> pydantic.BaseModel

    ReflectionVerdict --|> pydantic.BaseModel

    NodeSpec --|> pydantic.BaseModel

    ConditionSpec --|> pydantic.BaseModel

    EdgeSpec --|> pydantic.BaseModel

    TopologySpec --|> pydantic.BaseModel

    SubagentState *-- AnyMessage

    SubagentState *-- add_messages

    IncrementalSynthesis *-- MetadataResponse

    ValidationResult *-- ValidationStatus

    SearchResult *-- Section

    TopologySpec *-- NodeSpec

    TopologySpec *-- ConditionSpec

    TopologySpec *-- EdgeSpec
