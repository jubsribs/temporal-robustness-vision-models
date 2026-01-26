from pathlib import Path

BASE_DIR = Path("data/daily_raw/scripts-to-data-collection")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/results")

CAMERAS = [
    "camera_alpha",
    "camera_beta",
    "webcam_usb",
    "camera_sensor"
]

SENSORS = {
    "grove_gas_mq2_sensor": ["average_gas"],
    "grove_light_sensor_v1_2":["average_light"],
    "grove_loudness_sensor":["average_loudness"],
    "grove_dht11_sensor": [
        "average_temperature",
        "average_humidity"
    ],
    "grove_ir_thermal_sensor": [
        "average_object",
        "average_ambient"
    ],
    "grove_ultrasonic_ranger_sensor":["average_distance"]
}

SAMPLING_MINUTES = 10
