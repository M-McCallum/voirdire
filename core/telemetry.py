"""OpenTelemetry setup. Traces are OTel spans everywhere, never backend-native formats."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

DEFAULT_ENDPOINT = "http://localhost:4318"
SERVICE_NAME = "voirdire"


def setup_tracing(endpoint: str | None = None) -> TracerProvider:
    """Install a TracerProvider exporting OTLP/HTTP to the collector."""
    base = (endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or DEFAULT_ENDPOINT).rstrip(
        "/"
    )
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces")))
    trace.set_tracer_provider(provider)
    return provider


def tracer() -> trace.Tracer:
    return trace.get_tracer(SERVICE_NAME)
