
import time
import sys
import subprocess
import json
import logging
from datetime import datetime, timezone

import obd
from crewai.tools import tool
from serial.tools import list_ports

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. FIND AVAILABLE OBD / BLUETOOTH PORTS
# --------------------------------------------------

OBD_TIMEOUT = 2
OBD_BAUDRATE = 38400
PREFERRED_PORT = "COM7"
PROBE_TIMEOUT = 15  # hard limit per port, in seconds

# Runs in a child process so a hung port can be killed.
_PROBE_SNIPPET = """
import sys, obd
c = obd.OBD(portstr=sys.argv[1], baudrate=int(sys.argv[2]),
            timeout=int(sys.argv[3]), fast=False)
ok = c.is_connected()
c.close()
sys.exit(0 if ok else 1)
"""


# --------------------------------------------------
# 1. FIND AVAILABLE OBD / BLUETOOTH PORTS
# --------------------------------------------------
def get_candidate_ports():
    """Try the preferred port first, then every other detected port."""
    ports = list(list_ports.comports())

    if not ports:
        raise RuntimeError(
            "No OBD device detected: no serial ports are available. "
            "Check Bluetooth pairing and Device Manager."
        )

    return sorted(
        ports,
        key=lambda p: (
            p.device.upper() != PREFERRED_PORT.upper(),
            "OBD" not in p.description.upper(),
            "BLUETOOTH" not in p.description.upper(),
        ),
    )


# --------------------------------------------------
# 2. PROBE EACH PORT SAFELY, THEN CONNECT
# --------------------------------------------------
def probe_port(port_name):
    """Return (ok, reason). Never blocks longer than PROBE_TIMEOUT."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SNIPPET,
             port_name, str(OBD_BAUDRATE), str(OBD_TIMEOUT)],
            capture_output=True,
            timeout=PROBE_TIMEOUT,  # kills the child on timeout
        )
        if proc.returncode == 0:
            return True, "connected"
        return False, "no vehicle response"
    except subprocess.TimeoutExpired:
        return False, f"timed out after {PROBE_TIMEOUT}s (port hung)"
    except Exception as exc:
        return False, f"error: {exc}"


def find_obd_connection():
    """Check every port; return (connection, port_name) for the first that works."""
    results = {}

    for port_info in get_candidate_ports():
        port_name = port_info.device
        print(
            f"Trying OBD port: {port_name} ({port_info.description})", flush=True)

        ok, reason = probe_port(port_name)
        results[port_name] = reason
        print(f"  -> {reason}", flush=True)

        if not ok:
            continue

        time.sleep(1)  # let the adapter release the port after the probe

        try:
            connection = obd.OBD(
                portstr=port_name,
                baudrate=OBD_BAUDRATE,
                timeout=OBD_TIMEOUT,
                fast=False,
            )
            if connection.is_connected():
                return connection, port_name
            connection.close()
            results[port_name] = "probe ok, but real connection failed"
        except Exception as exc:
            results[port_name] = f"real connection error: {exc}"

    summary = "; ".join(f"{p}: {r}" for p, r in results.items())
    raise RuntimeError(
        f"Unable to connect to the vehicle. Results per port -> {summary}. "
        "Check ignition, Bluetooth pairing, and other apps using the adapter."
    )
# --------------------------------------------------
# 3. LIVE SENSOR COMMANDS
# --------------------------------------------------


LIVE_COMMANDS = {
    "engine_rpm": "RPM",
    "vehicle_speed": "SPEED",
    "coolant_temperature": "COOLANT_TEMP",
    "engine_load": "ENGINE_LOAD",
    "throttle_position": "THROTTLE_POS",
    "relative_throttle_position": "RELATIVE_THROTTLE_POS",
    "absolute_throttle_position_b": "ABSOLUTE_THROTTLE_POS_B",
    "absolute_load": "ABSOLUTE_LOAD",
    "intake_manifold_pressure": "INTAKE_PRESSURE",
    "intake_air_temperature": "INTAKE_TEMP",
    "ambient_air_temperature": "AMBIANT_AIR_TEMP",
    "mass_air_flow": "MAF",
    "barometric_pressure": "BAROMETRIC_PRESSURE",
    "control_module_voltage": "CONTROL_MODULE_VOLTAGE",
    "engine_runtime": "RUN_TIME",
    "fuel_level": "FUEL_LEVEL",
    "fuel_pressure": "FUEL_PRESSURE",
    "fuel_rail_pressure": "FUEL_RAIL_PRESSURE_DIRECT",
    "commanded_equivalence_ratio": "EQUIV_RATIO",
    "short_term_fuel_trim_bank_1": "SHRTFT1",
    "long_term_fuel_trim_bank_1": "LONGFT1",
    "short_term_fuel_trim_bank_2": "SHRTFT2",
    "long_term_fuel_trim_bank_2": "LONGFT2",
    "commanded_egr": "COMMANDED_EGR",
    "egr_error": "EGR_ERROR",
    "evaporative_purge": "EVAPORATIVE_PURGE",
    "warmups_since_dtc_clear": "WARMUPS_SINCE_DTC_CLEAR",
    "distance_since_dtc_clear": "DISTANCE_SINCE_DTC_CLEAR",
    "distance_with_mil_on": "DISTANCE_W_MIL",
    "time_since_dtc_cleared": "TIME_SINCE_DTC_CLEARED",
    "ethanol_percentage": "ETHANOL_PERCENT",
    "fuel_type": "FUEL_TYPE",
    "obd_compliance": "OBD_COMPLIANCE",
    "engine_reference_torque": "ENGINE_REF_TORQUE",
}


# --------------------------------------------------
# 4. DIAGNOSTIC AND FREEZE-FRAME COMMANDS
# --------------------------------------------------

DIAGNOSTIC_COMMANDS = {
    "mil_and_dtc_status": "STATUS",
    "confirmed_trouble_codes": "GET_DTC",
    "freeze_frame_trouble_codes": "FREEZE_DTC",
    "freeze_frame_engine_rpm": "FRPM",
    "freeze_frame_vehicle_speed": "FSPEED",
    "freeze_frame_coolant_temperature": "FCOOLANT",
    "freeze_frame_engine_load": "FLOAD",
    "freeze_frame_intake_pressure": "FINTAKE_PRESSURE",
    "freeze_frame_intake_temperature": "FINTAKE_TEMP",
    "freeze_frame_maf": "FMAF",
    "freeze_frame_throttle_position": "FTHROTTLE",
}


# --------------------------------------------------
# 5. OXYGEN SENSOR COMMANDS
# --------------------------------------------------

OXYGEN_COMMANDS = {
    "o2_sensor_bank1_sensor1": "O2_B1S1",
    "o2_sensor_bank1_sensor2": "O2_B1S2",
    "o2_sensor_bank2_sensor1": "O2_B2S1",
    "o2_sensor_bank2_sensor2": "O2_B2S2",
}


# --------------------------------------------------
# 6. NORMALIZE VALUES FOR JSON
# --------------------------------------------------

def normalize_value(value):
    if value is None:
        return None

    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return {
            "value": value.magnitude,
            "unit": str(value.units),
        }

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]

    return str(value)


# --------------------------------------------------
# 7. READ COMMANDS
# --------------------------------------------------

def read_commands(connection, command_map):
    results = {}
    unavailable = []

    for label, command_name in command_map.items():
        command = getattr(obd.commands, command_name, None)

        if command is None:
            results[label] = None
            unavailable.append({
                "parameter": label,
                "reason": "Command unavailable in installed python-OBD",
            })
            continue

        try:
            if not connection.supports(command):
                results[label] = None
                unavailable.append({
                    "parameter": label,
                    "reason": "Vehicle ECU does not report support",
                })
                continue
        except Exception:
            # If support detection is unavailable, try querying directly.
            pass

        try:
            response = connection.query(command)

            if response.is_null():
                results[label] = None
                unavailable.append({
                    "parameter": label,
                    "reason": "No usable response from vehicle",
                })
            else:
                results[label] = normalize_value(response.value)

        except Exception as exc:
            logger.warning("Query failed for %s: %s", label, exc)

            results[label] = None
            unavailable.append({
                "parameter": label,
                "reason": "Query failed",
            })

    return results, unavailable


# --------------------------------------------------
# 8. MAIN CREWAI TOOL
# --------------------------------------------------

@tool("collect_comprehensive_obd_ii_diagnostic_data")
def collect_obd_diagnostic_data(request: str) -> str:
    """
    Collect available OBD-II live data, trouble codes,
    freeze-frame readings, and oxygen sensor readings.
    Returns JSON and never clears trouble codes.
    """

    connection = None
    port_name = None

    try:
        connection, port_name = find_obd_connection()

        logger.warning("Collecting vehicle data on %s", port_name)

        live, live_unavailable = read_commands(
            connection, LIVE_COMMANDS
        )

        diagnostics, diagnostic_unavailable = read_commands(
            connection, DIAGNOSTIC_COMMANDS
        )

        oxygen, oxygen_unavailable = read_commands(
            connection, OXYGEN_COMMANDS
        )

        result = {
            "status": "success",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "connection": {
                "port": port_name,
                "protocol": connection.protocol_name(),
            },
            "live_data": live,
            "diagnostic_data": diagnostics,
            "oxygen_sensors": oxygen,
            "collection_summary": {
                "live_parameters_requested": len(LIVE_COMMANDS),
                "live_parameters_available": sum(
                    value is not None for value in live.values()
                ),
                "diagnostic_parameters_requested": len(
                    DIAGNOSTIC_COMMANDS
                ),
                "diagnostic_parameters_available": sum(
                    value is not None for value in diagnostics.values()
                ),
                "oxygen_parameters_requested": len(OXYGEN_COMMANDS),
                "oxygen_parameters_available": sum(
                    value is not None for value in oxygen.values()
                ),
            },
            "unavailable_data": (
                live_unavailable
                + diagnostic_unavailable
                + oxygen_unavailable
            ),
            "interpretation_limits": [
                "Missing data does not prove a fault or healthy operation.",
                "A trouble code alone does not prove a component has failed.",
                "Use vehicle-specific service information for diagnosis.",
                "Manufacturer-specific modules may not be accessible.",
            ],
        }

        return json.dumps(result, indent=2)

    except Exception as exc:
        logger.exception("OBD collection failed")

        return json.dumps({
            "status": "connection_error",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "connection": {"port": port_name},
            "message": str(exc),
            "live_data": {},
            "diagnostic_data": {},
            "oxygen_sensors": {},
        }, indent=2)

    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logger.exception("Could not close OBD connection")


# --------------------------------------------------
# 9. STANDALONE TEST
# --------------------------------------------------

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)

#     print("Testing OBD connection and collecting data...")

#     output = collect_obd_diagnostic_data.run(
#         request="Collect current OBD-II readings and diagnostic codes."
#     )

#     print(output)
