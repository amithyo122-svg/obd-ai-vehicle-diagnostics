# This tool searches public search results and returns links and snippets.
# It does not log into social accounts or access private posts.


import json
import re
from urllib.parse import urlparse

from ddgs import DDGS
from crewai.tools import tool


@tool("search_vehicle_community_content")
def search_vehicle_community_content(
    vehicle_json: str,
    obd_report: str,
) -> str:
    """
    Find public YouTube, Reddit, Facebook, Instagram and
    automotive forum results related to a vehicle's symptoms/DTCs.
    Returns titles, links and search snippets.
    """

    try:
        vehicle = json.loads(vehicle_json)
    except json.JSONDecodeError:
        return "Error: vehicle_json must be valid JSON."

    required = ["make", "model", "year", "engine"]
    if any(not str(vehicle.get(k, "")).strip() for k in required):
        return "Error: make, model, year and engine are required."

    codes = sorted(set(re.findall(
        r"\b[PCBU][0-3][0-9A-F]{3}\b", obd_report.upper()
    )))

    vehicle_name = (
        f"{vehicle['make']} {vehicle['model']} "
        f"{vehicle['year']} {vehicle['engine']}"
    )
    code_text = " ".join(codes)

    queries = [
        f'"{vehicle_name}" {" ".join(codes)}' if codes
        else f'"{vehicle_name}" engine symptoms',
        f'site:youtube.com/watch "{vehicle["make"]}" '
        f'"{vehicle["model"]}" {code_text or "diagnostic repair"}',
        f'site:reddit.com "{vehicle["make"]}" '
        f'"{vehicle["model"]}" {code_text or "engine problem"}',
        f'"{vehicle_name}" diagnostic repair forum',
        f'site:facebook.com "{vehicle["make"]}" '
        f'"{vehicle["model"]}" {code_text or "repair"}',
        f'site:instagram.com "{vehicle["make"]}" '
        f'"{vehicle["model"]}" {code_text or "repair"}',
    ]

    results, seen = [], set()
    errors = []

    try:
        with DDGS() as search:
            for query in queries:
                try:
                    for item in search.text(query, max_results=4) or []:
                        url = item.get("href") or item.get("url") or ""
                        if not url or url in seen:
                            continue

                        seen.add(url)
                        host = (urlparse(url).hostname or "").lower()
                        host = host.removeprefix("www.")

                        if "youtube.com" in host or "youtu.be" in host:
                            platform = "YouTube"
                        elif "reddit.com" in host:
                            platform = "Reddit"
                        elif "facebook.com" in host:
                            platform = "Facebook"
                        elif "instagram.com" in host:
                            platform = "Instagram"
                        else:
                            platform = "Automotive website/forum"

                        results.append({
                            "platform": platform,
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("body", ""),
                        })

                except Exception as exc:
                    errors.append(f"{query}: {exc}")

    except Exception as exc:
        return f"Community search failed: {exc}"

    return json.dumps({
        "vehicle": vehicle_name,
        "fault_codes": codes,
        "results": results[:20],
        "search_errors": errors[:4],
        "limitations": [
            "Search results may contain only titles and snippets.",
            "A result does not prove the reported fix applies to this vehicle.",
            "Private or non-indexed posts may not be discoverable.",
            "Do not claim to have watched videos or read full posts unless done.",
        ],
    }, indent=2)
