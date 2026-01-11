"""
resource_monitor.py
Collects system telemetry used by the deployment agent.
Uses psutil; on some boards temperature or power sensors may not be present.
"""

import psutil
import time
import platform

def get_telemetry():
    """
    Returns a dictionary with CPU %, available RAM (MB), temperature (C),
    and basic system metadata for UI and logging.
    """

    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    available_ram_mb = mem.available / (1024.0 ** 2)
    temp = None

    # Try real temperature sensors (if available)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for _, entries in temps.items():
                if entries:
                    temp = entries[0].current
                    break
    except Exception:
        temp = None

    # Fallback heuristic if sensor not present
    if temp is None:
        temp = 35.0 + (cpu / 100.0) * 40.0

    telemetry = {
        "cpu_percent": round(cpu, 2),
        "available_ram_mb": round(available_ram_mb, 2),
        "temperature_c": round(temp, 2),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "timestamp": time.time()
    }

    return telemetry

if __name__ == "__main__":
    print(get_telemetry())
