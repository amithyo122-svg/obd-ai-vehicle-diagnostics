
from crewai import Task
from agents.diagnostic_research_agent import diagnostic_research_agent
from tasks.obd_task1 import vehicle_monitor_task

diagnostic_research_task = Task(
    description="""
    Investigate the reported vehicle problem using the vehicle
    details and the output of the preceding OBD monitoring task.

    VEHICLE PROFILE:
    {vehicle_json}

    Use the monitoring task output provided in your context.

    TOOL WORKFLOW:

    STEP 1 — DOWNLOAD DOCUMENTS
    Call download_vehicle_documents first.
    Pass vehicle_json and the complete OBD monitoring report
    obtained from the task context.

    STEP 2 — RETRIEVE TECHNICAL EVIDENCE
    Call retrieve_vehicle_knowledge.
    Pass the same vehicle_json and OBD monitoring report.
    Review accepted/rejected documents and retrieved passages.

    STEP 3 — COMMUNITY RESEARCH
    Call search_vehicle_community_content.
    Search for relevant videos, Reddit discussions, public posts,
    and automotive forum results. Include actual returned links only.

    STEP 4 — COMBINE THE EVIDENCE
    Report:
    A. Vehicle and reported symptoms
    B. Technical documents and retrieved passages
    C. Community videos and discussions with links
    D. Possible causes supported by evidence
    E. Recommended verification checks
    F. Missing information and uncertainty

    EVIDENCE RULES:
    - Separate technical evidence from owner experiences.
    - Cite document filenames and PDF page numbers when available.
    - Never invent specifications, fixes, results, or links.
    - A trouble code alone does not prove a component has failed.
    - Recommend verification before replacing components.
    """,

    expected_output="""
    A structured automotive diagnostic research report containing
    technical evidence, actual community links, possible causes,
    verification checks, and limitations.
    """,

    agent=diagnostic_research_agent,

    context=[vehicle_monitor_task],
)
