from crewai import Task
from agents.final_diagnosis_agent import final_diagnosis_agent

from tasks.obd_task1 import vehicle_monitor_task
from tasks.diagnostic_research_task import diagnostic_research_task

final_diagnosis_task = Task(
    description="""
    Generate the final vehicle diagnosis using the results of
    the preceding monitoring and diagnostic research tasks.

    Review the previous task outputs and provide:
    1. Probable fault
    2. Supporting evidence
    3. Severity
    4. Recommended repair
    5. Further tests required
    """,
    expected_output=(
        "A complete evidence-based diagnostic report with possible "
        "causes, supporting evidence, next tests, severity, and "
        "confidence level."
    ),

    agent=final_diagnosis_agent,

    context=[
        vehicle_monitor_task,
        diagnostic_research_task,
    ],
)
