
import os
from dotenv import load_dotenv

from crewai import Agent, LLM

from tools.pdf_downloader import download_vehicle_documents
from tools.retrieve_vehicle_knowledge import retrieve_vehicle_knowledge
from tools.community_search_tool import search_vehicle_community_content

ollama_llm = LLM(
    model="ollama/qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

# llm = LLM(
#     model="groq/openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.2
# )


diagnostic_research_agent = Agent(
    role="Automotive Diagnostic Research Specialist",

    goal=(
        "Research vehicle faults using relevant service documents, "
        "retrieved technical evidence, and real-world community reports."
    ),

    backstory=(
        "You are an automotive research specialist supporting a "
        "vehicle diagnostic system. You investigate reported DTCs "
        "and symptoms using downloaded service documents, retrieval-"
        "augmented generation, YouTube videos, Reddit discussions, "
        "social posts and automotive forums. You separate verified "
        "technical documentation from anecdotal owner experiences. "
        "You report source links, filenames and page numbers where "
        "available, acknowledge missing evidence, and never claim "
        "a component has failed without sufficient evidence."
    ),

    llm=ollama_llm,

    tools=[
        download_vehicle_documents,
        retrieve_vehicle_knowledge,
        search_vehicle_community_content,
    ],

    verbose=True,
    allow_delegation=False,
    max_iter=6,
)
