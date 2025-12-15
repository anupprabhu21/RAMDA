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
    Returns a dictionary with CPU %, available RAM (MB), and a temperature reading (C).
    Temperature may be approximated if no sensor present.
    """
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    available_ram_mb = mem.available / (1024.0 ** 2)
    temp = None

    # try to get sensor readings; may not be available on all systems
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            # pick first sensor reading available
            for name, entries in temps.items():
                if entries:
                    temp = entries[0].current
                    break
    except Exception:
        temp = None

    # fallback temperature estimate if sensor not available
    if temp is None:
        # heuristic: base 35 + cpu_fraction * 0.5
        temp = 35.0 + (cpu / 100.0) * 40.0

    telemetry = {
        "cpu_percent": cpu,
        "available_ram_mb": available_ram_mb,
        "temperature_c": temp,
        "timestamp": time.time()
    }
    return telemetry

if __name__ == "__main__":
    print(get_telemetry())
