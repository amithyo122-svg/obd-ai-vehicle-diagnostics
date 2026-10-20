# This tool checks the downloaded documents, extracts their text page by page,
# labels vehicle-specific versus general DTC references, indexes eligible
# passages in ChromaDB, and retrieves the top five matches.


import hashlib
import json
import re
from pathlib import Path

import chromadb
from crewai.tools import tool
from pypdf import PdfReader
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_ROOT = PROJECT_ROOT / "data" / "vehicle_documents"
CHROMA_PATH = PROJECT_ROOT / "data" / "vehicle_chroma"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_")[:100]


def extract_codes(text: str) -> list[str]:
    return sorted(set(re.findall(
        r"\b[PCBU][0-3][0-9A-F]{3}\b", text.upper()
    )))


def classify_document(
    filename: str,
    full_text: str,
    vehicle: dict,
    codes: list[str],
) -> tuple[str, str]:
    content = (filename + " " + full_text).lower()

    make = str(vehicle["make"]).lower().strip()
    model = str(vehicle["model"]).lower().strip()
    year = str(vehicle["year"]).strip()
    engine = str(vehicle["engine"]).lower().strip()

    # Conservative first-pass match; this is not proof of applicability.
    if (
        make and model
        and make in content
        and model in content
        and (
            (year and year in content)
            or (engine and engine in content)
        )
    ):
        return "vehicle_specific_candidate", "Vehicle identity terms matched"

    matching_codes = [
        code for code in codes
        if re.search(
            rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])",
            content.upper(),
        )
    ]

    if matching_codes:
        return (
            "general_dtc_reference",
            "Matching DTC(s): " + ", ".join(matching_codes),
        )

    return "rejected", "No sufficient vehicle or DTC match"


def split_chunks(text: str, size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size].strip()

        if len(chunk) >= 50:
            chunks.append(chunk)

        if start + size >= len(text):
            break

        start += size - overlap

    return chunks


def get_collection(vehicle: dict):
    key = safe_name(
        f"{vehicle['make']}_{vehicle['model']}_"
        f"{vehicle['year']}_{vehicle['engine']}"
    ).lower()

    return client.get_or_create_collection(
        name=key,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


@tool("retrieve_vehicle_knowledge")
def retrieve_vehicle_knowledge(
    vehicle_json: str,
    obd_report: str,
) -> str:
    """
    Filter downloaded PDFs for relevance, index eligible pages
    in ChromaDB, and retrieve passages relevant to the OBD report.
    """

    try:
        vehicle = json.loads(vehicle_json)
    except json.JSONDecodeError:
        return "Error: vehicle_json must be valid JSON."

    required = ["make", "model", "year", "engine"]
    if any(not str(vehicle.get(k, "")).strip() for k in required):
        return "Error: make, model, year and engine are required."

    codes = extract_codes(obd_report)
    vehicle_key = (
        f"{vehicle['make']}_{vehicle['model']}_"
        f"{vehicle['year']}_{vehicle['engine']}"
    )
    folder = DOCUMENT_ROOT / safe_name(vehicle_key)

    if not folder.exists():
        return (
            f"No document folder found: {folder}. "
            "Run download_vehicle_documents first."
        )

    collection = get_collection(vehicle)
    accepted, rejected = [], []
    ids, texts, metadatas = [], [], []

    for pdf_path in folder.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf_path))

            if reader.is_encrypted:
                rejected.append(f"{pdf_path.name}: encrypted PDF")
                continue

            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((page_number, text))

            full_text = "\n".join(text for _, text in pages)

            if not full_text.strip():
                rejected.append(f"{pdf_path.name}: no extractable text")
                continue

            category, reason = classify_document(
                pdf_path.name, full_text, vehicle, codes
            )

            if category == "rejected":
                rejected.append(f"{pdf_path.name}: {reason}")
                continue

            accepted.append(
                f"{pdf_path.name}: {category}; {reason}"
            )

            for page_number, page_text in pages:
                for chunk_number, chunk in enumerate(split_chunks(page_text)):
                    raw_id = (
                        f"{pdf_path.resolve()}|{page_number}|{chunk_number}"
                    )
                    doc_id = hashlib.sha1(
                        raw_id.encode("utf-8")
                    ).hexdigest()

                    ids.append(doc_id)
                    texts.append(chunk)
                    metadatas.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "category": category,
                    })

        except Exception as exc:
            rejected.append(f"{pdf_path.name}: {exc}")

    if ids:
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    if collection.count() == 0:
        return json.dumps({
            "status": "no_eligible_documents",
            "accepted": accepted,
            "rejected": rejected,
        }, indent=2)

    query = (
        f"Vehicle: {vehicle['make']} {vehicle['model']} "
        f"{vehicle['year']}, engine {vehicle['engine']}.\n"
        f"Fault codes: {', '.join(codes) or 'none reported'}.\n"
        "Find diagnostic procedures, code descriptions, specifications, "
        "possible causes and verification tests.\n"
        f"OBD report:\n{obd_report}"
    )

    result = collection.query(
        query_texts=[query],
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    retrieved = []
    for i, passage in enumerate(passages):
        retrieved.append({
            "source": metas[i].get("source"),
            "page": metas[i].get("page"),
            "category": metas[i].get("category"),
            "distance": distances[i] if i < len(distances) else None,
            "text": passage,
        })

    return json.dumps({
        "vehicle": vehicle,
        "fault_codes": codes,
        "accepted_documents": accepted,
        "rejected_documents": rejected,
        "retrieved_passages": retrieved,
        "limitations": [
            "Text matching is a first-pass relevance check, not proof of applicability.",
            "General DTC references are not vehicle-specific procedures.",
            "Review document provenance and vehicle compatibility.",
            "Do not infer a failed component from a DTC alone.",
        ],
    }, indent=2)
