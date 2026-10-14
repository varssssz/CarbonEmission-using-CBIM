from datetime import date

from backend.reconciliation.engine import run_reconciliation
from backend.reconciliation.inputs import Record, ReconciliationInput
from backend.reconciliation.models import SourceRef

start = date(2026, 1, 1)
end = date(2026, 1, 31)

cases = [
    ("Negative value", -100.0),
    ("NaN", float("nan")),
    ("Positive infinity", float("inf")),
    ("Negative infinity", float("-inf")),
]

for name, value in cases:
    reported = Record(
        category="electricity",
        value=value,
        unit="kWh",
        period_start=start,
        period_end=end,
        source=SourceRef("report.csv", "row 1"),
    )

    evidence = Record(
        category="electricity",
        value=100.0,
        unit="kWh",
        period_start=start,
        period_end=end,
        source=SourceRef("evidence.csv", "row 1"),
    )

    try:
        findings = run_reconciliation(
            ReconciliationInput(
                reported=[reported],
                evidence=[evidence],
            )
        )

        print(f"\n{name}:")
        for finding in findings:
            print(
                f"  Rule: {finding.rule_id}, "
                f"Reported: {finding.reported_value}, "
                f"Evidence: {finding.evidence_value}, "
                f"Difference: {finding.difference}"
            )

        if not findings:
            print("  No findings generated")

    except Exception as error:
        print(f"\n{name}: ERROR - {type(error).__name__}: {error}")