---
title: SokoFlow
emoji: 🧩
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: "5.29.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# SokoFlow

CPU diffusion demo for 8×8 Sokoban. A puzzle is on the board at load; **Play** runs diffusion (≤2 clicks from load; shipped as 1).

Colorful pieces (wall / floor / box / goal / player / box-on-goal). Chrome is cream, not a Gradio control stack.

Headline eval is **scramble-hard**, not the historical GS-T5 20.8%. Microban OOD is a separate table.

Repo source of truth: `gradio_app/` in [sivaratrisrinivas/sokoflow](https://github.com/sivaratrisrinivas/sokoflow).
