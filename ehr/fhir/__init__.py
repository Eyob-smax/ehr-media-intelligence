"""Task 2: map the cleaned dataset to HL7 FHIR R4 resources, bundle, validate and store them.

fhir.resources 8.x (the Pydantic v2 line) ships R5 as its default package and
R4B under ``fhir.resources.R4B``. R4B is a minor, backwards-compatible revision
of R4 for every resource used here (Patient, Encounter, DocumentReference,
DiagnosticReport, Observation, Bundle), so we validate against R4B models and
emit plain R4-shaped JSON.
"""
