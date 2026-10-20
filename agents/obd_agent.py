
import os
from dotenv import load_dotenv

from crewai import Agent, LLM

from tools.obd_tool import collect_obd_diagnostic_data


ollama_llm = LLM(
    model="ollama/qwen3:8b",
    temperature=0.1,
)

# llm = LLM(
#     model="groq/openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.2
# )


vehicle_monitor_agent = Agent(
    role="Vehicle Monitoring and Fault Detection Specialist",

    goal=(
        "Collect comprehensive OBD-II vehicle data, identify "
        "reported trouble codes, detect readings that may need "
        "attention, and distinguish measured evidence from "
        "unverified assumptions."
    ),

    backstory=(
        "You are an automotive diagnostic specialist working "
        "with live OBD-II sensor readings and ECU diagnostic codes. "
        "You understand engine parameters, emissions data, freeze-frame "
        "information, and diagnostic trouble codes. You never assume "
        "that a trouble code proves a specific component has failed. "
        "You identify missing readings, explain uncertainty, and "
        "recommend further checks based on the available evidence."
    ),

    llm=ollama_llm,

    tools=[
        collect_obd_diagnostic_data,
    ],

    verbose=True,
    allow_delegation=False,
    max_iter=3,
)
