"""
SecurePredict V0.7 smoke test.

This checks that the frozen model artifacts load and that the inference
functions accept the exact schemas exposed by those artifacts.

It deliberately uses synthetic values; it does not touch the test set.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Make "src" importable when run as:
# python -X utf8 src\inference\smoke_test.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import get_expected_features, predict_ai4i, predict_rt_iot2022


def main() -> None:
    print("SECUREPREDICT V0.7 - INFERENCE SMOKE TEST")
    print("=" * 60)

    schemas = get_expected_features()

    ai4i_features = schemas["ai4i"]
    rt_features = schemas["rt_iot2022"]

    print(f"AI4I expected features: {len(ai4i_features)}")
    print(f"RT-IoT2022 expected features: {len(rt_features)}")

    ai4i_sample = {
        "Type": "L",
        "Air temperature": 298.2,
        "Process temperature": 308.5,
        "Rotational speed": 1500.0,
        "Torque": 42.0,
        "Tool wear": 120.0,
    }

    ai4i_result = predict_ai4i(ai4i_sample)

    print("\nAI4I sample prediction")
    print(f"  Failure probability : {ai4i_result['failure_probability']:.4f}")
    print(f"  Primary status      : {ai4i_result['primary_status']}")
    print(f"  Early warning       : {ai4i_result['early_warning_status']}")

    # The RT-IoT final model uses numeric flow features only.
    # Zero-filled values here are for interface testing only, not a benchmark.
    rt_sample = {feature: 0.0 for feature in rt_features}
    rt_result = predict_rt_iot2022(pd.DataFrame([rt_sample]))

    print("\nRT-IoT2022 synthetic interface test")
    print(f"  Attack probability : {rt_result['attack_probability']:.4f}")
    print(f"  Classification     : {rt_result['classification']}")
    print(f"  Risk level         : {rt_result['risk_level']}")

    print("\n" + "=" * 60)
    print("V0.7 INFERENCE SMOKE TEST PASSED")
    print("No test-set rows were loaded or evaluated.")


if __name__ == "__main__":
    main()
