'''
Collection file for all classes to serve as basemodels or to structure output and their methods (if any).
'''

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, get_origin, get_args, Union

from agentic_dd.agentlib.nodes_factory import register_structured_output

CATEGORIES = (
    "Archival Materials", "Civil Society and Interest Groups", "Corruption", "Economic History and Economics",
    "Education", "Elections and Referendums", "Elites", "GDR Research", "Journalistic and Social media",
    "Migration and Displacement", "Miscellaneous", "Political Parties and Politicians", "Protests", "Public Opinion",
    "Repression", "Security", "Social Policy"
)

@register_structured_output("MetadataResponse")
class MetadataResponse(BaseModel):
    # Dataset metadata -> not all are inferable from the data itself, e.g., the uploader likely is not; exclude here
    Title: Optional[str] = Field(default=None, description="Short but concise title for the dataset")
    Subtitle: Optional[str] = None
    Creators: Optional[str] = None
    Creator_type: Optional[Literal["Person", "Institution"]] = None
    Institutional_affiliation: Optional[str] = None

    Main_Category: Optional[Literal[CATEGORIES]] = None
    Additional_Categories: Optional[List[Literal[CATEGORIES]]] = None

    Publication_date: Optional[str] = Field(default=None, description="Format YYYY-MM-DD")
    Date_period_of_data_creation_from: Optional[str] = Field(default=None, description="Format YYYY-MM-DD")
    Date_period_of_data_creation_to: Optional[str] = Field(default=None, description="Format YYYY-MM-DD")
    Date_of_data_creation_text: Optional[str] = None
    Time_period_covered_in_data_collection_from: Optional[str] = None
    Time_period_covered_in_data_collection_to: Optional[str] = None
    Time_period_covered_text: Optional[str] = None

    Sources_of_data: Optional[List[str]] = None
    Data_types: Optional[List[Literal[
        "archival documents", "audio", "images", "interviews",
        "statistics", "survey", "text document", "video", "other"
    ]]] = None
    Datatype_text: Optional[str] = None

    Countries: Optional[List[Literal[
        "Armenia", "Azerbaijan", "Belarus", "Estonia", "GDR", "Georgia",
        "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Moldova",
        "Other", "Russia", "Tajikistan", "Turkmenistan", "Ukraine", "USSR", "Uzbekistan"
    ]]] = None
    Languages: Optional[List[str]] = None
    Disciplines: Optional[List[str]] = None
    Keywords: Optional[List[str]] = None

    Related_dataset: Optional[str] = None
    Related_dataset_text: Optional[str] = None
    Related_publications: Optional[str] = Field(default=None, description="Includes subtitle(s), separated by '.'")
    Related_projects: Optional[List[str]] = None

    Funding: Optional[str] = None
    Methods_of_data_collection: Optional[List[str]] = None
    Methods_of_data_analysis: Optional[List[str]] = None


class PlanItem(BaseModel):
    task: str


@register_structured_output("RouterDecision")
class RouterDecision(BaseModel):
    route: str
    reasoning: str = Field(description="Briefly justify your routing decision.")
    confidence: float = Field(ge=0.0, le=1.0)

    plan: Optional[List[PlanItem]] = None   # spawn route only


class SummaryEval(BaseModel):
    faithfulness_score: int = Field(..., ge=1, le=5, description="Score from 1-5 for faithfulness")
    conciseness_score: int = Field(..., ge=1, le=5, description="Score from 1-5 for conciseness")
    coverage_score: int = Field(..., ge=1, le=5, description="Score from 1-5 for coverage")
    
    reasoning: str = Field(..., description="Short explanation for the scores")
    
    divergence_examples: str = Field(
        default="No major divergences.", 
        description="Examples of where the generated text significantly diverges from the ground truth"
    )

@register_structured_output("IncrementalSynthesis")
class IncrementalSynthesis(BaseModel):
    updated_summary: str = Field(description="Full updated running abstract, superseding the previous version")
    updated_metadata: MetadataResponse
    notes: str = Field(default="", description="What the new source added, confirmed, or contradicted")

@register_structured_output("ReflectionVerdict")
class ReflectionVerdict(BaseModel):
    sufficient: bool = Field(description="True if enough information has been gathered to finalize")
    conflicts_detected: bool = Field(default=False, description="True if sources disagree on some fact")
    reasoning: str


# Mirror field types from the structured output validation class
def get_metadata_field_categories(model_class):
    ''' for the current metadata types this suffices, if we introduced tuple, set, etc. it breaks '''
    list_fields = []
    text_fields = []
    
    for field_name, field_info in model_class.model_fields.items():
        # Get the type annotation (e.g., List[str] or Optional[List[str]])
        annotation = field_info.annotation
        
        # Check if the field is a List
        # We check if the origin is list, or if it's Optional and the inner arg is list
        is_list = False
        
        if get_origin(annotation) is list:
            is_list = True
        elif get_origin(annotation) is Union: # Handle Optional[List[X]]
            # Optional is Union[X, None]
            args = get_args(annotation)
            if args and get_origin(args[0]) is list:
                is_list = True
        
        if is_list:
            list_fields.append(field_name)
        else:
            # Everything else (str, int, Literal, etc.) is treated as text/exact match
            text_fields.append(field_name)
            
    return list_fields, text_fields