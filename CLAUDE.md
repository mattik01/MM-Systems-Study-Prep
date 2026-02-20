# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a study preparation repository for a **Multimodal Systems** university course taught by Gualtiero Volpe (Casa Paganini / InfoMus / DIBRIS, University of Genoa). It contains lecture materials, project notes, and a poster generator script. There is no build system or test suite.

## Contents

- `full-MM-Systems-Slides.pdf` — full set of lecture slides
- `full-MM-systems-speechscript.txt` — comprehensive narrated speech script covering all lecture topics (structured by slide numbers)
- `full-MM-systems-speechscript-tts.mp3` — TTS audio rendering of the speech script
- `project/` — Project 14 (critical reading of Niknazar & Mednick, IEEE TPAMI 2024)
  - `projects.pdf` — list of all available course projects
  - `project-selection.txt` — selected project (Project 14)
  - `project14-plan.md` — full critical reading plan with deliverables checklist
  - `study-notes.md` — structured study notes (problem, architecture, results, course connections, exam Q&A)
  - `project14-poster.pptx` — generated A0 poster
  - `make_poster.py` — Python script that generates the poster; edit the `CONTENT` dict to update poster text
  - `papers/` — reference papers directory (main paper PDF, AASM manual, reading list)

## Poster Generation

```bash
python project/make_poster.py
```
Requires `python-pptx`. Outputs `project/project14-poster.pptx`. Edit the `CONTENT` dict at the top of `make_poster.py` to change poster content.

## Course Topics Covered

The speech script covers the full lecture series in order:

1. **Introduction** — WIMP paradigm, Post-WIMP systems, multimodal processing pipeline (input → representation → prediction → mapping → output), history ("Put That There", 1980)
2. **Designing a Multimodal System** — motivation for multimodality, modality selection principles, complementarity, environment adaptation
3. **Input modalities** — sensory channels, fusion strategies (early/late/hybrid), interpretation
4. **Movement analysis** — kinematic features (velocity, acceleration, curvature), Quantity of Motion (QoM), kinetic energy, anthropometric tables
5. **Gesture taxonomy** — McNeill's framework (iconic, metaphoric, deictic, beat gestures), expressive gestures
6. **Gesture unitizing and annotation** — thresholding, mid-level feature computation

## Working with This Repo

When asked to help study or explain concepts, refer to `full-MM-systems-speechscript.txt` as the primary source — it is more detailed than the slides PDF and includes formulas, examples, and elaborations keyed to slide numbers.

For project-related work: `project/project14-plan.md` is the master plan with all deliverables and checklists. `project/study-notes.md` contains the detailed analysis. Sections marked *(verify)* in study notes are reconstructed from model knowledge and need checking against the actual paper PDF in `project/papers/main-paper.pdf`.
