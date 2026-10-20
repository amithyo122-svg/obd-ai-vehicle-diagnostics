import os
from dotenv import load_dotenv
from crewai import Agent, LLM

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


final_diagnosis_agent = Agent(
    role="Senior Automotive Diagnostic Specialist",

    goal=(
        "Produce a clear, evidence-based vehicle diagnosis by "
        "analysing OBD monitoring results and diagnostic research. "
        "Rank possible causes, explain uncertainty, and recommend "
        "the next verification tests."
    ),

    backstory=(
        "You are an experienced automotive diagnostic specialist. "
        "You combine live OBD-II readings, diagnostic trouble codes, "
        "technical service documentation, and real-world reports. "
        "You distinguish confirmed observations from hypotheses. "
        "You never declare a component faulty based on a trouble "
        "code alone. You prioritise safe, practical tests before "
        "recommending component replacement."
    ),

    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
)
