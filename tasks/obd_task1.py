
from crewai import Task

from agents.obd_agent import vehicle_monitor_agent

vehicle_monitor_task = Task(
    description="""
    Monitor the vehicle using the following vehicle information:

    {vehicle_json}

    Analyze the available OBD data and identify abnormalities.
    Produce a structured monitoring report.
    """,
    expected_output="A structured report of OBD readings and detected abnormalities.",
    agent=vehicle_monitor_agent,
)
