
# This searches for candidate PDFs using the vehicle profile and
# reported fault codes. It saves them in a predictable folder that the RAG tool
#  will use.


import ipaddress
import json
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

import requests
from ddgs import DDGS
from crewai.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_ROOT = PROJECT_ROOT / "data" / "vehicle_documents"
MAX_PDF_BYTES = 15 * 1024 * 1024


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_")[:100]


def vehicle_folder(vehicle: dict) -> Path:
    key = (
        f"{vehicle['make']}_{vehicle['model']}_"
        f"{vehicle['year']}_{vehicle['engine']}"
    )
    return DOCUMENT_ROOT / safe_name(key)


def public_https_url(url: str) -> bool:
    """Reject non-HTTPS URLs and hosts resolving to non-public IPs."""
    try:
        parsed = urlparse(url)

        if parsed.scheme != "https" or not parsed.hostname:
            return False

        if parsed.username or parsed.password:
            return False

        addresses = socket.getaddrinfo(
            parsed.hostname, 443, type=socket.SOCK_STREAM
        )

        return bool(addresses) and all(
            ipaddress.ip_address(item[4][0]).is_global
            for item in addresses
        )
    except (ValueError, OSError):
        return False


def download_pdf(url: str, destination: Path) -> str:
    """Download a PDF with redirect, size and content checks."""
    current_url = url

    with requests.Session() as session:
        for _ in range(4):
            if not public_https_url(current_url):
                raise ValueError("URL is not an allowed public HTTPS URL")

            response = session.get(
                current_url,
                timeout=(5, 20),
                allow_redirects=False,
                stream=True,
                headers={"User-Agent": "VehicleResearchBot/1.0"},
            )

            if response.is_redirect:
                location = response.headers.get("Location")
                response.close()

                if not location:
                    raise ValueError("Redirect has no destination")

                from urllib.parse import urljoin
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            declared_size = response.headers.get("Content-Length")

            if declared_size and int(declared_size) > MAX_PDF_BYTES:
                response.close()
                raise ValueError("PDF exceeds size limit")

            chunks = []
            total = 0

            try:
                for chunk in response.iter_content(65536):
                    if not chunk:
                        continue

                    total += len(chunk)

                    if total > MAX_PDF_BYTES:
                        raise ValueError("PDF exceeds size limit")

                    chunks.append(chunk)
            finally:
                response.close()

            content = b"".join(chunks)

            if not content.startswith(b"%PDF-"):
                raise ValueError(
                    f"Not a PDF file (content type: {content_type})"
                )

            destination.write_bytes(content)
            return current_url

    raise ValueError("Too many redirects")


@tool("download_vehicle_documents")
def download_vehicle_documents(
    vehicle_json: str,
    obd_report: str,
) -> str:
    """
    Search for and download public PDF service documents.

    vehicle_json: JSON with make, model, year and engine.
    obd_report: OBD monitoring report with fault codes and symptoms.
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

    folder = vehicle_folder(vehicle)
    folder.mkdir(parents=True, exist_ok=True)

    queries = [
        f'"{vehicle_name}" service manual PDF',
        f'"{vehicle_name}" workshop manual diagnostic PDF',
    ]

    if codes:
        queries.append(
            f'"{vehicle_name}" {" ".join(codes)} diagnostic PDF'
        )

    downloaded = []
    failed = []
    seen = set()

    try:
        with DDGS() as search:
            for query in queries:
                try:
                    results = search.text(query, max_results=6)

                    for item in results or []:
                        url = item.get("href") or item.get("url") or ""

                        if not url or url in seen:
                            continue

                        seen.add(url)
                        parsed = urlparse(url)
                        title = item.get("title", "document")

                        # Skip search-engine redirects and obvious ads.
                        if parsed.hostname and (
                            parsed.hostname.endswith("bing.com")
                            or parsed.hostname.endswith("google.com")
                        ):
                            continue

                        # Avoid downloading arbitrary HTML pages.
                        # This tool only accepts direct PDF-like URLs.
                        if not parsed.path.lower().endswith(".pdf"):
                            continue

                        filename = safe_name(title)[:75] + ".pdf"
                        destination = folder / filename

                        if destination.exists():
                            downloaded.append({
                                "title": title,
                                "file": str(destination),
                                "status": "already downloaded",
                            })
                            continue

                        try:
                            final_url = download_pdf(url, destination)

                            downloaded.append({
                                "title": title,
                                "file": str(destination),
                                "url": final_url,
                                "status": "downloaded; relevance not yet verified",
                            })

                        except Exception as exc:
                            failed.append(f"{url}: {exc}")

                        if len(downloaded) >= 5:
                            break

                except Exception as exc:
                    failed.append(f"Search failed for {query}: {exc}")

                if len(downloaded) >= 5:
                    break

    except Exception as exc:
        return f"Search service failed: {exc}"

    return json.dumps({
        "vehicle": vehicle_name,
        "folder": str(folder),
        "downloaded_documents": downloaded,
        "failed": failed[:10],
        "next_step": (
            "Call retrieve_vehicle_knowledge to evaluate document "
            "relevance and retrieve supporting passages."
        ),
        "warning": (
            "Downloaded files are unverified candidates, not confirmed "
            "vehicle-specific service documentation."
        ),
    }, indent=2)
