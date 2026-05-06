from copy import deepcopy
from src.domain.schema import ICC_CANONICAL_TEMPLATE


class ICCTranslator:
    def __init__(self):
        pass

    def to_icc(self, record: dict) -> dict:
        icc = deepcopy(ICC_CANONICAL_TEMPLATE)

        # 🔹 Product
        icc["product"]["hs_code"] = record.get("hs_code")
        icc["product"]["description"] = record.get("product_description")
        icc["product"]["category"] = record.get("product_family")

        # 🔹 Origin
        icc["origin"]["country"] = record.get("declared_origin_country")
        icc["origin"]["export_country"] = record.get("country_of_export")
        icc["origin"]["last_transformation_country"] = record.get(
            "country_of_last_substantial_transformation"
        )

        # 🔹 Trade
        icc["trade"]["quantity"] = record.get("shipment_quantity")
        icc["trade"]["unit"] = record.get("quantity_unit")
        icc["trade"]["value"] = record.get("shipment_value_usd")
        icc["trade"]["currency"] = "USD"
        icc["trade"]["incoterm"] = record.get("incoterm")

        # 🔹 Transport
        icc["transport"]["mode"] = record.get("transport_mode")
        icc["transport"]["port_of_export"] = record.get("port_of_export")
        icc["transport"]["port_of_entry"] = record.get("port_of_entry")

        # 🔹 Emissions
        icc["emissions"]["value"] = record.get(
            "embedded_emissions_tco2e_per_tonne"
        )
        icc["emissions"]["estimation_method"] = record.get(
            "emissions_estimation_method"
        )
        icc["emissions"]["electricity_source"] = record.get(
            "electricity_source_tag"
        )
        icc["emissions"]["recycled_content_percent"] = record.get(
            "recycled_content_percent"
        )

        # 🔹 Identifiers
        icc["identifiers"]["shipment_id"] = record.get("shipment_id")
        icc["identifiers"]["invoice_id"] = record.get("invoice_id")
        icc["identifiers"]["exporter_id"] = record.get("exporter_id")
        icc["identifiers"]["importer_id"] = record.get("importer_id")

        # 🔹 Traceability
        icc["traceability"]["supplier_chain_depth"] = record.get(
            "supplier_chain_depth"
        )
        icc["traceability"]["traceability_score"] = record.get(
            "traceability_completeness_score"
        )
        icc["traceability"]["batch_reference_available"] = record.get(
            "batch_lot_reference_available"
        )
        icc["traceability"]["production_site_available"] = record.get(
            "production_site_identifier_available"
        )

        # 🔹 Compliance
        icc["compliance"]["origin_certificate_available"] = record.get(
            "origin_certificate_available"
        )
        icc["compliance"]["origin_certificate_match_score"] = record.get(
            "origin_certificate_match_score"
        )
        icc["compliance"]["document_consistency_score"] = record.get(
            "document_consistency_score"
        )
        icc["compliance"]["supporting_document_count"] = record.get(
            "supporting_document_count"
        )

        # 🔹 Risk
        icc["risk"]["risk_label"] = record.get("risk_label")
        icc["risk"]["rule_flag_count"] = record.get("rule_flag_count")
        icc["risk"]["anomaly_count"] = record.get("anomaly_count")

        # 🔹 Context (Buy / Ship / Pay heuristic)
        icc["context"] = self._infer_context(record)

        return icc

    def _infer_context(self, record: dict) -> str:
        if record.get("invoice_id"):
            return "buy"
        elif record.get("shipment_id"):
            return "ship"
        return "unknown"