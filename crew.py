
import json

from crewai import Crew, Process
import crewai.llms.cache as crewai_cache

from tasks.final_diagnosis_task import final_diagnosis_task
from tasks.diagnostic_research_task import diagnostic_research_task
from tasks.obd_task1 import vehicle_monitor_task

crewai_cache.mark_cache_breakpoint = lambda msg: msg


def run_diagnosis(vehicle_details, obd_data):
    """
    Run CrewAI using the vehicle details and real OBD
    readings collected by Streamlit.
    """

    workflow_input = {
        "vehicle": vehicle_details,
        "obd_data": obd_data,
    }

    crew = Crew(
        agents=[
            vehicle_monitor_task.agent,
            diagnostic_research_task.agent,
            final_diagnosis_task.agent,
        ],
        tasks=[
            vehicle_monitor_task,
            diagnostic_research_task,
            final_diagnosis_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff(
        inputs={
            "vehicle_json": json.dumps(workflow_input)
        }
    )
