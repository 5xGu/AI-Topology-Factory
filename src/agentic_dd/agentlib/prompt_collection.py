'''
A simple collection file for (system) prompts and prompt templates.
'''
import json
from agentic_dd.agentlib.structuredOutputs import get_metadata_field_categories, MetadataResponse
from agentic_dd.agentlib.prompts import register_prompt
from textwrap import dedent
from typing import List, Dict, Any

# ---- DEFAULT prompt collection ---- #
DEFAULT_PROMPTS = {
    "summarizer_default": "Write a descriptive (technical) summary of the input, that is at most 300 words long.",
    "metadata_default": "Extract metadata from the following input. Return ONLY a valid JSON object.",
    "worker_default": "You are a worker agent executing one subtask.",
    "agent_default": "You are the top-level agent. Use delegation for complex subtasks.",
    "subagent_default": "You are a delegated sub-agent. Complete the given task.",
    "critic_default": "Critique the aggregated results below for consistency and gaps.",
    "merge_default": "Merge the following outputs into a single coherent answer.",
    "assistant_default": "You are a helpful assistant.",
}

# ---- ORCHESTRATION prompt collection ---- #
# NOTE: {allowed_routes} and {tool_names} are rendered per-call via
# PromptSpec.variables (see prompts.render_template) and populated by
# router.build_router_compute at invoke time. 

ORCHESTRATION_PROMPTS = {
    "free_default": dedent("""
        You are a routing controller for a data-analysis system implemented in LangGraph. You
        should receive basic information on files from a dataset and decide how to best process them. Typical input are
        filenames and filesizes. The filenames are often structured to include information about the category they are from
        in their dataset. For example:
            - somedataset/data/data.json,
            - somedataset/metadata/metadata.csv
            - somedataset/description_PDF/description.json

        Mind that the files or filepaths you or the system receive are all normalized to be .csv and .json files to
        allow uniform processing. Their category was preserved.

        Based on this information, decide how the system is best routed to achieve these goals:
            - Summarize the dataset
            - Classify the dataset into specified metadata categories

        Your response's `route` field must be exactly one of the following, currently allowed for this run: {allowed_routes}

        If 'agent' or 'spawn' is among the allowed routes and you choose it, the tools available to the
        resulting agent/worker(s) are fixed in advance and are not something you choose; for reference,
        the tools registered in this system are: {tool_names}

        Leave fields not required empty.
    """).strip(),
}

# ---- AGENT prompt collection ---- #
AGENT_PROMPTS = {
    "selective_file_reading_addendum": dedent("""
        The listed files are not necessarily meant to all be read and combined. Directory
        names indicate role/category (e.g. .../data/*, .../metadata/*, .../description/*).
        For a group of many similar files (e.g. several CSVs of the same kind), reading one
        or a small representative sample is usually sufficient; prefer reading any dedicated
        description/metadata files in full. Only call read_file on paths you actually need.
    """).strip(),
}

# ---- EXTRACTION prompt collection ---- #
EXTRACTION_PROMPTS = {
    "llm_extractor_default": dedent("""
        You are an information extraction assistant. Given the raw content of a data file
        (and any accompanying documentation), extract the key structured facts that would
        help downstream summarization and metadata classification:
        - main topic/subject
        - geographic and temporal scope
        - data type or collection method
        - key entities/organizations named
        - any dates mentioned

        Present these as a concise bulleted list of facts actually present in the text.
        Do not summarize, interpret, or draw conclusions -- extract only explicit facts.
    """).strip(),
}

# ---- METADATA prompt collection ---- #
_LIST_FIELDS, _TEXT_FIELDS = get_metadata_field_categories(MetadataResponse)
_ALL_METADATA_FIELDS = ", ".join(_LIST_FIELDS + _TEXT_FIELDS)

METADATA_PROMPTS = {
    "metadata_extraction_expert": dedent(f"""
        You are a metadata extraction expert. You recognize metadata fields, even if they are separated by white spaces or linebreaks. Extract all (!) following metadata fields from the provided text.
        If a field is not present, return it as `None`. Do not add anything else to such fields, besides 'None'.

        Example output format:
        'Title': 'Example Title',
        'Subtitle': None,
        'Countries': ['Ukraine'],
        'Keywords': ['corruption', 'Ukraine'],
        'Languages': ['English'],
        'Disciplines': ['Political Science'],
        'Creators': 'Institute',
        'Date_period_of_data_creation_from': '2007',
        'Date_period_of_data_creation_to': '2024',
        'Sources_of_data': 'survey',
        'Data_types': 'quantitative',
        'Publication_date': 'March 18 2025',
        'Related_publications': ['publication1', 'publication2']

        Return the output as a JSON object with the following keys: {_ALL_METADATA_FIELDS}.
    """).strip()
}

# ---- METADATA EXEMPLAR collection ---- #
FEWSHOT_EXEMPLARS: List[Dict[str, Any]] = [
    {
        "Title": "Protests Armenia 2015: Video Database",
        "Subtitle": None,
        "Creators": "Research Centre for East European Studies at the University of Bremen",
        "Creator_type": "Institution",
        "Institutional_affiliation": "Research Centre for East European Studies at the University of Bremen",
        "Main_Category": "Protests",
        "Additional_Categories": ["Social Policy"],
        "Publication_date": "Sept. 18, 2025, 2:48 p.m.",
        "Date_period_of_data_creation_from": "March 9, 2020",
        "Date_period_of_data_creation_to": "May 1, 2020",
        "Date_of_data_creation_text": "March to May 2020",
        "Time_period_covered_in_data_collection_from": "June 22, 2015",
        "Time_period_covered_in_data_collection_to": "Aug. 29, 2019",
        "Time_period_covered_text": "June 2015 mainly, few till August 2019 (posting date)",
        "Sources_of_data": ["YouTube"],
        "Data_types": ["other", "video"],
        "Datatype_text": "excel database",
        "Countries": ["Armenia"],
        "Languages": ["Russian", "English", "Armenian"],
        "Disciplines": ["Comparative Politics", "Political Science"],
        "Keywords": ["2015", "Armenia", "Electric Yerevan", "Electromaidan", "Protest", "Yerevan"],
        "Related_dataset": None,
        "Related_dataset_text": "https://doi.org/10.48320/CB1F2D84-C2D0-4A6C-BB26-EE7D8E55C7F8",
        "Related_publications": None,
        "Related_projects": [
            "This data collection has been produced as part of the research project 'Comparing protest "
            "actions in Soviet and post-Soviet spaces', which is organised by the Research Centre for "
            "East European Studies at the University of Bremen with financial support from the "
            "Volkswagen Foundation."
        ],
        "Funding": "additional own contribution to project",
        "Methods_of_data_collection": ["Keyword Search", "Snow Ball Search"],
        "Methods_of_data_analysis": ["Raw Data", "Raw Data For Analysis"],
    },
    {
        "Title": "Needs Assessment for USAID Independent Living Program 2021",
        "Subtitle": None,
        "Creators": "Caucasus Research Resource Centers",
        "Creator_type": "Institution",
        "Institutional_affiliation": "Caucasus Research Resource Centers",
        "Main_Category": "Social Policy",
        "Additional_Categories": ["Public Opinion"],
        "Publication_date": "Jan. 9, 2026, 11:57 a.m.",
        "Date_period_of_data_creation_from": "Sept. 9, 2021",
        "Date_period_of_data_creation_to": "Sept. 16, 2021",
        "Date_of_data_creation_text": "September 9, 2021 - September 16, 2021",
        "Time_period_covered_in_data_collection_from": "Sept. 9, 2021",
        "Time_period_covered_in_data_collection_to": "Sept. 16, 2021",
        "Time_period_covered_text": "September 9, 2021 - September 16, 2021",
        "Sources_of_data": ["Caucasus Research Resource Centers"],
        "Data_types": ["survey"],
        "Datatype_text": None,
        "Countries": ["Georgia"],
        "Languages": ["English", "Georgian"],
        "Disciplines": ["Social Sciences", "Sociology"],
        "Keywords": [
            "Georgia", "Independent Living Program", "Needs As- sessment",
            "People With Disabilities", "People With Disabilities Issues",
            "Social Needs", "Usaid",
        ],
        "Related_dataset": None,
        "Related_dataset_text": None,
        "Related_publications": None,
        "Related_projects": None,
        "Funding": (
            "This survey was conducted in partnership with the McLain Association (MAC) Georgia for "
            "Children and the Coalition for Independent Living (CIL), with the financial support of USAID."
        ),
        "Methods_of_data_collection": ["Cati  (Computer-Assisted  Telephone  Interviews)", "Quantitative Survey"],
        "Methods_of_data_analysis": None,
    },
    {
        "Title": "How Ukrainians assess reforms, corruption and civic activism in October 2022",
        "Subtitle": None,
        "Creators": "Sologoub, Dorontseva",
        "Creator_type": "Person",
        "Institutional_affiliation": None,
        "Main_Category": "Public Opinion",
        "Additional_Categories": ["Corruption"],
        "Publication_date": "Jan. 3, 2023, 12:44 p.m.",
        "Date_period_of_data_creation_from": "2022-10-12",
        "Date_period_of_data_creation_to": "2022-10-30",
        "Date_of_data_creation_text": "The dataset was created in November-December 2022.",
        "Time_period_covered_in_data_collection_from": "2022-10-12",
        "Time_period_covered_in_data_collection_to": "2022-10-30",
        "Time_period_covered_text": "The data was collected between 12-30 October 2022.",
        "Sources_of_data": [
            "The survey was implemented by Info Sapiens polling company during 12-30 October 2022; "
            "method: online survey, sample: 2113 respondents 18+ who lived in government-controlled "
            "areas as of February 23rd, including those who moved abroad after February 24th; the "
            "sample is representative for Ukraine's population by sex, age, settlement size and region "
            "according to Ukrstat data as of 01.01.22, theoretical error is no more than 2.2% for total sample."
        ],
        "Data_types": ["survey"],
        "Datatype_text": None,
        "Countries": ["Ukraine"],
        "Languages": ["Ukrainian"],
        "Disciplines": ["Comparative Politics", "Sociology"],
        "Keywords": ["Bribery", "Civic Activism", "Civil Society", "Corruption", "Reforms"],
        "Related_dataset": None,
        "Related_dataset_text": None,
        "Related_publications": (
            "Ilona Sologoub, Yelizaveta Dorontseva, Reforms, corruption and civic activism: opinion of "
            "Ukrainians in October-2022, 2022, "
            "https://voxukraine.org/en/reforms-corruption-and-civic-activism-opinion-of-ukrainians-in-october-2022/"
        ),
        "Related_projects": ["Support of think tanks, https://www.irf.ua/en/program/support-of-think-tanks/."],
        "Funding": (
            "This publication was produced within the framework of the 'Support of think tanks' project "
            "which is carried out by the International Renaissance Foundation with the financial support "
            "of the Embassy of Sweden in Ukraine."
        ),
        "Methods_of_data_collection": ["Online Interview", "Online Questionnaire"],
        "Methods_of_data_analysis": ["Statistical Analysis"],
    },
]


def _metadata_prompt_with_exemplars(n: int) -> str:
    base = METADATA_PROMPTS["metadata_extraction_expert"]
    chosen = FEWSHOT_EXEMPLARS[:n]
    if not chosen:
        return base
    block = "\n\n".join(
        f"Example output {i + 1}:\n{json.dumps(ex, indent=2, ensure_ascii=False)}"
        for i, ex in enumerate(chosen)
    )
    return base + "\n\nHere are worked examples of correct extractions:\n\n" + block


for _n in (0, 1, 3):
    register_prompt(f"metadata_extraction_expert_{_n}shot", _metadata_prompt_with_exemplars(_n))

# ---- SUMMARY prompt collection ---- #
SUMMARY_PROMPTS = {
    "summarizer_dataset_abstract": dedent("""
        You are writing the descriptive abstract field for a research data repository catalogue entry.
        A researcher browsing the catalogue will read this abstract to get an intuitive grasp on the dataset's contents and its relevance for their
        own work. This is their first impression of the dataset and decides how likely they are to further investigate the files within it.

        In 2-4 sentences per item in the following list, cover what is discernible from the input:
        - what the dataset is about (topic/subject)
        - its geographic and temporal scope, if stated
        - how the data was collected or its source/type, if stated
        - the main contributions of the dataset

        Do not include information that is not present in or reasonably inferable from the input.
        Do not open with generic phrases like "This dataset..." or "This document describes...";
        write directly about the subject matter. Stick with what is actually contained within the dataset, do not make up information.
    """).strip(),

    "summarizer_constrained": dedent("""
        Summarize the input in a short descriptive abstract, maximum 300 words total.
        Do not use hedging language (e.g. "seems to", "appears to", "may").
        Do not reference the act of summarizing or describe the document itself; stick to the information
        actually contained in the input.
        State the content directly.
    """).strip(),

    "summarizer_cot": dedent("""
        Before writing the summary, first identify for yourself: (1) the main topic,
        (2) the geographic/temporal scope if present, (3) the data source or method if present, (4) the main contributions of the dataset,
        Then write a at most 300 words descriptive summarizing abstract on the dataset.

        Output ONLY the final abstract. Do not include your intermediate
        identification step, labels, or any text other than the summary itself.
    """).strip(),

    "summary_eval": dedent("""
        You are an expert abstract editor. Your task is to evaluate the quality of an AI generated descriptive abstract. Please evaluate the quality of the generated summary based on the original text.

        Please rate the summary on a scale of 1-5 for:
        1. Faithfulness: Does the summary contain only information supported by the original text? (No hallucinations)
            - If information is not contained in the ground truth but can genuinely be infered from it, this is okay. For example, if the ground truth mentions an institution by name, but does not state explicitly that it is an institution,
            it is okay for the model to diverge and do so.
        2. Conciseness: Is it brief and to the point?
        3. Coverage: Does it capture the main points of the ground truth?

        Please produce examples of where the generated text diverges from the ground truth in a major way if there are any of that. As you do not have access to the information the abstract is based on,
        do not attempt a review of factuality of the contents. Flag only cases where the produced abstract significantly diverges, e.g. contradicts or mentions information seemingly contradicting the ground
        truth.

        Output your answer in JSON format:
        {{
        "faithfulness_score": <int>,
        "conciseness_score": <int>,
        "coverage_score": <int>,
        "reasoning": "<short explanation>",
        "divergence_examples": <some example of divergence between ground truth and generated summary>"
        }}

        Generated Summary:
        {generated_summary}

        Ground Truth Summary (Reference):
        {ground_truth}
    """).strip()
}

ITERATIVE_PROMPTS = {
    "incremental_synthesis_default": dedent("""
        You are incrementally building a dataset summary and metadata classification,
        one source file at a time. You will be shown the current working summary and
        metadata (possibly empty, if this is the first source), followed by a new
        source's content. Update the summary and metadata to incorporate the new
        source: add missing information, confirm existing claims, and correct or
        flag anything the new source contradicts. Do not discard prior information
        unless the new source directly contradicts it -- in that case, prefer the
        more specific/authoritative source and note the discrepancy.
    """).strip(),
    "reflection_gate_default": dedent("""
        You are reviewing a working dataset summary and metadata draft, built so far
        from a subset of available source files. Decide whether enough information
        has been gathered to finalize, or whether further sources should be examined
        first. Flag explicitly if any gathered sources appear to disagree with each
        other on a factual point.
    """).strip(),
}


def register_all_prompts():
    for name, template in {
        **DEFAULT_PROMPTS,
        **SUMMARY_PROMPTS,
        **METADATA_PROMPTS,
        **ORCHESTRATION_PROMPTS,
        **EXTRACTION_PROMPTS,
        **AGENT_PROMPTS,
        **ITERATIVE_PROMPTS,
    }.items():
        register_prompt(name, template)


register_all_prompts()