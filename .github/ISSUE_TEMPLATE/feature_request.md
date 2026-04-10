---
name: Feature / task
about: Propose a feature or track a concrete task
title: "feat: "
labels: []
assignees: []
---

## Goal

<!-- What should improve and why? -->

## Scope

<!-- What is in scope for this change? What is explicitly out of scope? -->

## deltai boundary

deltai is the AI brain layer only (routing, RAG, tools, APIs). Game telemetry ingestion belongs in external services that call **`POST /ingest`**, not inside core deltai as raw parsers or UDP listeners. Does this request respect that boundary? (yes / no / unsure)

## Acceptance criteria

- [ ]
- [ ]

## Notes

<!-- Optional: design sketch, dependencies, related issues -->
