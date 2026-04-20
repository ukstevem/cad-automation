"""Exceptions raised by the IFC parser stack.

All inherit from :class:`app.exceptions.CADAutomationException` so the router
layer's existing handlers catch them.
"""
from app.exceptions import CADAutomationException


class IFCHeaderError(CADAutomationException):
    """The IFC header is malformed or missing required fields."""


class IFCVendorNotSupportedError(CADAutomationException):
    """The IFC file was produced by an application we don't yet support."""


class IFCParseError(CADAutomationException):
    """A vendor-specific parser failed while processing the file."""
