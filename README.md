Help Clinicians with medical responses


# 🇰🇪 Kenya Clinical Reasoning Challenge — Zindi AI4Health

## 🩺 Can AI Mirror Human Clinical Reasoning in Kenya?

This project was submitted for the **Kenya Clinical Reasoning Challenge** hosted on [Zindi Africa](https://zindi.africa), in collaboration with [PATH](https://www.path.org/?utm_source=google&utm_medium=paid&utm_campaign=20980934625&utm_term=path&content=157984876469&gad_source=1&gad_campaignid=20980934625&gbraid=0AAAAAD3kOABYEYyzS6Yz0M_QNucrp5QDJ&gclid=CjwKCAjwkbzEBhAVEiwA4V-yqhVbEuJHNlb2QBgUjCR-3b4Kh2emdzKg_ctVW30dXxM19ebq8xKBthoCaGoQAvD_BwE) - A HeathTech NGO that focuses on automating healthcare infrastructure.

- This project is a unique and socially impactful task aiming to replicate how real frontline nurses in rural Kenya respond to complex, real-world clinical scenarios.

---

## 📍 Context

In underserved regions, **frontline healthcare workers** make rapid, critical decisions under pressure and with limited support. This challenge provides **400 expert-labeled clinical vignettes**—each combining:
- A nurse's professional background
- A complex patient scenario
- Facility context (hospital/dispensary/clinic)

The goal? **Predict the nurse’s actual response**, matching the judgment of trained human clinicians.

These vignettes span **maternal health, pediatrics, infectious diseases, critical care**, and more—offering a diverse, real-world clinical simulation.

---

## 🧠 Objective

Build a model that:
- Mimics clinician decision-making with contextual awareness
- Captures **Kenya-specific** constraints and challenges
- Maintains ethical sensitivity and practical applicability

---

## 🔧 Tools & Techniques

| Tool / Method | Description |
|---------------|-------------|
| **Ollama** | Lightweight LLM deployment engine for local inference |
| **Qwen 2.5B Instruct** | A strong open-source instruction-following LLM by Alibaba |
| **QLoRA** | Efficient fine-tuning technique for large models on consumer hardware |
| **Prompt Engineering** | Zero-shot prompt templates crafted to mimic real clinical reasoning |
| **LangChain** | For structured data routing and clinical context templating |
| **Excel** | Manual cleaning and normalization of nurse responses and case metadata |

---

## 🧪 Approach

1. **Data Preprocessing**:
   - Cleaned and standardized all responses using Excel.
   - Extracted and normalized context fields (nurse experience, facility level, etc.).

2. **Prompt Design**:
   - Used **zero-shot prompts** structured like:
     > _"You are a frontline nurse in a rural Kenyan clinic. Here's your patient's presentation..."_

3. **Fine-Tuning**:
   - Fine-tuned Qwen 2.5B using **QLoRA** on a curated subset to align better with Kenyan healthcare nuances.

4. **Inference**:
   - Deployed the fine-tuned model via **Ollama** for efficient local inference.
   - Processed test samples through LangChain’s structured pipeline for consistent reasoning output.

---

## 📊 Results & Insights

- **Small Data, Big Impact**: Despite just 400 training samples, fine-tuning + prompt engineering captured critical reasoning nuances.
- **Local Relevance Matters**: Embedding contextual factors like facility type and nurse experience significantly improved alignment with human clinicians.
- **Open-Source + Efficiency**: QLoRA + Ollama proved powerful for socially impactful, low-resource AI tasks.

---

## 🌍 Why This Matters

> _"In places where decisions can't wait and resources are scarce, AI should assist—not replace—human care."_

This project reinforces the potential of **context-aware, locally adapted AI** to support—not override—healthcare professionals in under-resourced settings.

---


