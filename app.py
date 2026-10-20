
import json
import html
import queue
import threading
import time
import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="OBD Doctor",
    page_icon="🚗",
    layout="wide",
)


# =====================================================
# STYLING
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #e5e7eb;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
}

h1, h2, h3 {
    color: #f8fafc;
}

[data-testid="stMetric"] {
    background: #141c30;
    border: 1px solid #283650;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #a5b4cc;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
}

.panel {
    background: #121b2e;
    border: 1px solid #283650;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.agent-card {
    background: #10192b;
    border: 1px solid #283650;
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}

.agent-active {
    border-color: #38bdf8;
    background: #10243a;
}

.agent-done {
    border-color: #34d399;
}

.muted {
    color: #94a3b8;
    font-size: 13px;
}

.pulse {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #38bdf8;
    margin-right: 8px;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: .25; }
    100% { opacity: 1; }
}

.loader-track {
    height: 6px;
    background: #253149;
    border-radius: 10px;
    overflow: hidden;
}

.loader-bar {
    height: 100%;
    width: 35%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    animation: loading 1.3s infinite ease-in-out;
}

@keyframes loading {
    from { transform: translateX(-110%); }
    to { transform: translateX(310%); }
}

div.stButton > button {
    min-height: 46px;
    border-radius: 10px;
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background: #0e1628;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "obd_data": None,
    "diagnosis": None,
    "diagnosis_error": None,
    "agent_roles": [],
    "agent_steps": [],
    "activity_message": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# VEHICLE SELECTION
# =====================================================

with st.sidebar:
    st.title("🚗 OBD Doctor")
    st.caption("Vehicle diagnostics")
    st.divider()

    st.subheader("Vehicle details")

    car_model = st.selectbox(
        "Car model",
        [
            "Hyundai Creta",
            "Maruti Suzuki Swift",
            "Maruti Suzuki Wagon R",
            "Tata Nexon",
            "Tata Punch",
            "Mahindra XUV700",
            "Kia Seltos",
            "Honda City",
            "Toyota Innova",
            "Other",
        ],
    )

    car_year = st.selectbox(
        "Manufacturing year",
        list(range(2026, 1999, -1)),
        index=4,
    )

    fuel_type = st.selectbox(
        "Fuel type",
        ["Petrol", "Diesel", "CNG", "Hybrid", "Electric", "Unknown"],
    )

    odometer = st.number_input(
        "Odometer (km)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1000,
    )

    st.divider()

    if st.button("Clear readings and report"):
        st.session_state.obd_data = None
        st.session_state.diagnosis = None
        st.session_state.diagnosis_error = None
        st.session_state.agent_roles = []
        st.session_state.agent_steps = []
        st.session_state.activity_message = None
        st.rerun()


# =====================================================
# HEADER
# =====================================================

st.title("🚘 OBD Doctor")
st.caption("Vehicle telemetry and AI-assisted diagnostics")

st.markdown(
    f"**{car_model}** · {car_year} · {fuel_type}"
)

st.divider()


# =====================================================
# DATA HELPERS
# =====================================================

def parse_obd_result(raw):
    """Normalize the real OBD tool response for the dashboard."""

    if isinstance(raw, str):
        data = json.loads(raw)
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError("Unexpected OBD tool response.")

    if not isinstance(data, dict):
        raise ValueError("OBD tool must return a JSON object.")

    # Preserve the original tool output and its connection status.
    data["live_sensor_data"] = data.get(
        "live_data",
        data.get("live_sensor_data", {}),
    )

    diagnostic_data = data.get("diagnostic_data", {})

    if not isinstance(diagnostic_data, dict):
        diagnostic_data = {}

    # Extract the trouble codes from the actual diagnostic response.
    dtcs = diagnostic_data.get(
        "confirmed_trouble_codes",
        data.get("trouble_codes", []),
    )

    data["trouble_codes"] = dtcs if isinstance(dtcs, list) else []

    if not isinstance(data["live_sensor_data"], dict):
        data["live_sensor_data"] = {}

    return data


def display_value(value, unit=""):
    """Format returned values without inventing missing readings."""

    if value is None or value == "":
        return "N/A"

    # python-OBD values may already contain their measurement units.
    if isinstance(value, dict) and "value" in value:
        actual_value = value["value"]
        actual_unit = value.get("unit", "")
        return (
            "N/A"
            if actual_value is None
            else f"{actual_value} {actual_unit}".strip()
        )

    if isinstance(value, str) and value.strip().lower() in {
        "nan", "none", "null", "n/a", "na"
    }:
        return "N/A"

    return f"{value}{unit}"


def format_label(name):
    """Turn engine_rpm into Engine RPM."""

    labels = {
        "engine_rpm": "Engine RPM",
        "vehicle_speed_kmh": "Vehicle speed",
        "coolant_temperature_c": "Coolant temperature",
        "battery_voltage_v": "Battery voltage",
        "engine_load_percent": "Engine load",
        "throttle_position_percent": "Throttle position",
        "intake_air_temperature_c": "Intake air temperature",
        "intake_manifold_pressure_kpa": "Intake manifold pressure",
    }

    return labels.get(
        name,
        name.replace("_", " ").capitalize(),
    )


def format_unit(name):
    """Choose a display unit based on the sensor key."""

    units = {
        "engine_rpm": " rpm",
        "vehicle_speed_kmh": " km/h",
        "coolant_temperature_c": " °C",
        "battery_voltage_v": " V",
        "engine_load_percent": "%",
        "throttle_position_percent": "%",
        "intake_air_temperature_c": " °C",
        "intake_manifold_pressure_kpa": " kPa",
    }

    return units.get(name, "")


# =====================================================
# FETCH OBD DATA AND RUN CREWAI
# =====================================================


def run_diagnostic_worker(event_queue, vehicle_details):
    """
    Collect actual OBD readings, then run CrewAI with the
    same readings and vehicle details entered in Streamlit.
    """

    try:
        event_queue.put({
            "type": "status",
            "message": "Connecting to the OBD adapter...",
        })

        from tools.obd_tool import collect_obd_diagnostic_data

        # Collect real vehicle data.
        raw_obd = collect_obd_diagnostic_data.run(
            request=(
                "Collect currently available live OBD-II sensor "
                "readings, diagnostic trouble codes, and status."
            )
        )

        obd_data = parse_obd_result(raw_obd)

        # Stop if OBD collection or connection failed.
        if obd_data.get("status") != "success":
            raise RuntimeError(
                obd_data.get(
                    "message",
                    "OBD connection failed. Diagnosis cancelled."
                )
            )

        # Preserve the readings and attach user-entered details.
        obd_data["selected_vehicle"] = vehicle_details

        event_queue.put({
            "type": "obd_data",
            "data": obd_data,
        })

        event_queue.put({
            "type": "status",
            "message": "OBD data collected. Starting CrewAI agents...",
        })

        # Use the reusable function from the updated crew.py.
        from crew import run_diagnosis

        result = run_diagnosis(
            vehicle_details=vehicle_details,
            obd_data=obd_data,
        )

        event_queue.put({
            "type": "complete",
            "result": str(result),
        })

    except Exception as exc:
        event_queue.put({
            "type": "error",
            "message": f"{type(exc).__name__}: {exc}",
        })


# =====================================================
# DIAGNOSIS BUTTON
# =====================================================

st.subheader("Vehicle readings")

st.caption(
    "Readings appear after the data source returns them. "
    "Unavailable parameters are displayed as N/A."
)

run_button = st.button(
    "Run Vehicle Diagnosis",
    type="primary",
    use_container_width=True,
)


# =====================================================
# EXECUTE WORKFLOW
# =====================================================

if run_button:
    st.session_state.obd_data = None
    st.session_state.diagnosis = None
    st.session_state.diagnosis_error = None
    st.session_state.agent_roles = []
    st.session_state.agent_steps = []

    event_queue = queue.Queue()

    vehicle_details = {
        "model": car_model,
        "year": car_year,
        "fuel_type": fuel_type,
        "odometer_km": odometer,
    }

    worker = threading.Thread(
        target=run_diagnostic_worker,
        args=(event_queue, vehicle_details),
        daemon=True,
    )

    worker.start()

    status_placeholder = st.empty()
    agent_placeholder = st.empty()
    timer_placeholder = st.empty()

    current_message = "Preparing diagnostic workflow..."
    roles = []
    active_role = None
    completed_result = None
    error_message = None
    started_at = time.time()

    while worker.is_alive() or not event_queue.empty():

        while not event_queue.empty():
            event = event_queue.get()

            if event["type"] == "status":
                current_message = event["message"]

            elif event["type"] == "obd_data":
                st.session_state.obd_data = event["data"]

            elif event["type"] == "roles":
                roles = event["roles"]
                st.session_state.agent_roles = roles

            elif event["type"] == "agent_step":
                active_role = event["role"]

                step = {
                    "role": active_role,
                    "message": event["message"],
                }

                st.session_state.agent_steps.append(step)

                current_message = (
                    f'{active_role}: {event["message"]}'
                )

            elif event["type"] == "complete":
                completed_result = event["result"]

            elif event["type"] == "error":
                error_message = event["message"]

        elapsed = int(time.time() - started_at)

        status_placeholder.markdown(
            f"""
            <div class="panel">
                <span class="pulse"></span>
                <b>{html.escape(current_message)}</b>
                <p class="muted">{elapsed} seconds elapsed</p>
                <div class="loader-track">
                    <div class="loader-bar"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if roles:
            agent_html = ""

            for role in roles:
                safe_role = html.escape(str(role))

                if role == active_role:
                    css_class = "agent-active"
                    state = "Processing a reasoning step"
                elif any(
                    item["role"] == role
                    for item in st.session_state.agent_steps
                ):
                    css_class = "agent-done"
                    state = "Reasoning step observed"
                else:
                    css_class = ""
                    state = "Waiting for activity"

                agent_html += f"""
                <div class="agent-card {css_class}">
                    <b>{safe_role}</b>
                    <div class="muted">{state}</div>
                </div>
                """

            agent_placeholder.markdown(
                agent_html,
                unsafe_allow_html=True,
            )

        timer_placeholder.caption(
            "The workflow is running. Completion time depends on the model and tools."
        )

        time.sleep(0.25)

    if error_message:
        st.session_state.diagnosis_error = error_message

    elif completed_result is not None:
        st.session_state.diagnosis = completed_result

    # Refresh to display readings and report in their normal locations.
    st.rerun()


# =====================================================
# DISPLAY ACTUAL OBD READINGS
# =====================================================

obd_data = st.session_state.obd_data

if obd_data is None:
    st.info("Select your vehicle and click Run Vehicle Diagnosis to retrieve readings.")

else:
    sensor_data = obd_data.get("live_sensor_data", {})

    if sensor_data:
        # Only show keys returned by the OBD tool.
        items = list(sensor_data.items())

        for start in range(0, len(items), 4):
            cols = st.columns(4)

            for col, (name, value) in zip(
                cols,
                items[start:start + 4],
            ):
                with col:
                    st.metric(
                        format_label(name),
                        display_value(value, format_unit(name)),
                    )

    else:
        st.warning("The OBD tool returned no sensor readings.")

    st.divider()
    st.subheader("Diagnostic trouble codes")

    trouble_codes = obd_data.get("trouble_codes", [])

    if trouble_codes:
        if all(isinstance(code, dict) for code in trouble_codes):
            st.dataframe(
                pd.DataFrame(trouble_codes),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                pd.DataFrame({
                    "Trouble code": [
                        display_value(code)
                        for code in trouble_codes
                    ]
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.write("No trouble-code data returned.")

    # Show extra data returned by the tool, if present.
    excluded_keys = {
        "live_sensor_data",
        "trouble_codes",
        "selected_vehicle",
        "vehicle",
        "year",
        "fuel_type",
        "odometer_km",
        "data_source",
        "connection_status",
        "diagnostic_note",
    }

    extra_data = {
        key: value
        for key, value in obd_data.items()
        if key not in excluded_keys
    }

    if extra_data:
        with st.expander("Other returned OBD data"):
            st.json(extra_data)


# =====================================================
# FINAL AI REPORT
# =====================================================

if st.session_state.diagnosis_error:
    st.error("Diagnosis failed.")
    st.code(
        st.session_state.diagnosis_error,
        language="text",
    )

if st.session_state.diagnosis:
    st.divider()
    st.subheader("Final AI diagnosis")

    st.markdown(st.session_state.diagnosis)

    st.download_button(
        "Download diagnosis report",
        data=st.session_state.diagnosis,
        file_name="obd_diagnosis.txt",
        mime="text/plain",
    )
