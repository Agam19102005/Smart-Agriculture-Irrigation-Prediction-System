import random
from datetime import datetime

def generate_sensor_reading():
    """Return one simulated IoT sensor event. No hardware is accessed."""
    soil = round(random.uniform(8, 90), 1)
    temperature = round(random.uniform(18, 41), 1)
    humidity = round(random.uniform(25, 95), 1)
    rainfall = round(random.choices(
        [random.uniform(0, 1), random.uniform(1, 8), random.uniform(8, 25)],
        weights=[0.55, 0.35, 0.10], k=1
    )[0], 1)
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "soil_moisture": soil,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall_mm": rainfall,
    }
