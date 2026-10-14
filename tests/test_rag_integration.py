
import unittest
from unittest.mock import patch

from backend.rag.evidence import (
    build_guidance_query,
    enrich_finding,
    enrich_findings,
    explain_finding,
)
from backend.reconciliation.models import (
    Finding,
    FindingStatus,
    Severity,
    SourceRef,
)


def make_finding():
    return Finding(
        finding_id="test-finding-1",
        rule_id="R2",
        title="Inconsistent units for electricity",
        severity=Severity.MEDIUM,
        explanation="Records use different units.",
        recommended_action="Convert values to compatible units.",
        status=FindingStatus.OPEN,
        sources=[SourceRef(file="report.pdf", location="row 12")],
        uncertainty_notes="Units may be compatible after conversion.",
    )


def make_evidence():
    return {
        "text": "Guidance passage about purchased electricity.",
        "source_filename": "Scope 2 Guidance.pdf",
        "page_number": 12,
        "chunk_index": 0,
        "distance": 0.25,
    }


class TestRagEvidenceIntegration(unittest.TestCase):
    def test_query_uses_finding_details(self):
        query = build_guidance_query(make_finding())

        self.assertIn("R2", query)
        self.assertIn("Inconsistent units", query)
        self.assertIn("Convert values", query)

    @patch("backend.rag.evidence.search_guidance")
    def test_enrichment_preserves_original_finding_fields(self, search):
        finding = make_finding()
        original = finding.to_dict()
        search.return_value = [make_evidence()]

        result = enrich_finding(finding)

        for key, value in original.items():
            self.assertEqual(result[key], value)

        self.assertEqual(result["guidance_status"], "evidence_retrieved")
        self.assertEqual(
            result["guidance_evidence"][0]["source_filename"],
            "Scope 2 Guidance.pdf",
        )
        self.assertEqual(result["guidance_evidence"][0]["page_number"], 12)

        # Enrichment must not mutate the original object.
        self.assertEqual(finding.to_dict(), original)

    @patch("backend.rag.evidence.search_guidance")
    def test_no_evidence_is_reported_without_inventing_citations(self, search):
        search.return_value = []

        result = enrich_finding(make_finding())

        self.assertEqual(result["guidance_evidence"], [])
        self.assertEqual(result["guidance_status"], "no_relevant_evidence")
        self.assertIn("No guidance passages", result["review_assistance"]["guidance_summary"])

    def test_explanation_layer_discloses_limitations(self):
        finding = make_finding()

        result = explain_finding(finding, [make_evidence()])

        self.assertEqual(result["finding_summary"], finding.explanation)
        self.assertEqual(
            result["recommended_review"], finding.recommended_action
        )
        self.assertIn("not proof", result["interpretation_limit"])

    @patch("backend.rag.evidence.search_guidance")
    def test_multiple_findings_are_enriched(self, search):
        search.return_value = [make_evidence()]

        results = enrich_findings([make_finding(), make_finding()])

        self.assertEqual(len(results), 2)
        self.assertTrue(
            all(item["guidance_status"] == "evidence_retrieved" for item in results)
        )


if __name__ == "__main__":
    unittest.main()
