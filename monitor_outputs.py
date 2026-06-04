# =========================
# monitor_outputs.py
# Monitor des topics de sortie : metrics, anomalies, alerts
# =========================

import json
from kafka import KafkaConsumer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    METRICS_TOPIC,
    ANOMALIES_TOPIC,
    ALERTS_TOPIC,
)


consumer = KafkaConsumer(
    METRICS_TOPIC,
    ANOMALIES_TOPIC,
    ALERTS_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    api_version=(0, 10, 1),
    group_id="hospital-output-monitor",
    auto_offset_reset="latest",
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)


def main():
    print("Output monitor started.")
    print(f"Listening topics: {METRICS_TOPIC}, {ANOMALIES_TOPIC}, {ALERTS_TOPIC}")

    for message in consumer:
        topic = message.topic
        data = message.value

        print("\n" + "=" * 90)
        print(f"Topic={topic} | Partition={message.partition} | Offset={message.offset}")

        if topic == METRICS_TOPIC:
            print(
                f"METRICS | "
                f"Zone={data.get('zone_name')} | "
                f"Energy={data.get('zone_energy_kwh')} kWh | "
                f"Avg10={data.get('avg_energy_last_10_events')} | "
                f"Energy/Occupant={data.get('energy_per_occupant')} | "
                f"Risk={data.get('risk_score')} | "
                f"Status={data.get('status')}"
            )

        elif topic == ANOMALIES_TOPIC:
            print(
                f"ANOMALY | "
                f"Zone={data.get('zone_name')} | "
                f"Type={data.get('anomaly_type')} | "
                f"Severity={data.get('severity')} | "
                f"Value={data.get('value')} | "
                f"Message={data.get('message')}"
            )

        elif topic == ALERTS_TOPIC:
            print(
                f"ALERT | "
                f"Zone={data.get('zone_name')} | "
                f"Level={data.get('alert_level')} | "
                f"Type={data.get('anomaly_type')} | "
                f"Message={data.get('message')} | "
                f"Action={data.get('recommended_action')}"
            )


if __name__ == "__main__":
    main()