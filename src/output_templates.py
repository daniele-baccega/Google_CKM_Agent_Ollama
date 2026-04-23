"""Output templates for the CKM Syndrome Multi-Specialist Consultation.

This module defines standardized output formats:
1. Consultation Snapshot - Default compact output (≤250 words)
2. Peri-op Medication Stoplight Table - Expandable detailed table
3. Specialty Rationale - Expandable specialist reasoning
4. Citations - Expandable guideline references
"""

# Consultation Snapshot Template (default output - ≤250 words)
CONSULTATION_SNAPSHOT_TEMPLATE = """
## 📋 Consultation Snapshot

**A) One-Line Problem:**
{one_line_problem}

**B) 5 Key Facts:**
{key_facts}

**C) 5 Key Risks:**
{key_risks}

**D) Decisions Needed Today:**
{decisions_needed}

**E) Next Steps:**
{next_steps}

---
*Reply: **A** for peri-op medication stoplight table | **B** for specialty rationale | **C** for citations*
"""

# Peri-op Medication Stoplight Table Template
PERIOP_MEDICATION_TABLE_TEMPLATE = """
## 💊 Peri-op Medication Stoplight Table

| Medication | Continue | Hold | Restart Criteria | Owner / Guideline |
|------------|:--------:|:----:|------------------|-------------------|
{medication_rows}

---
*Reply: **B** for specialty rationale | **C** for citations | **Back** to return to snapshot*
"""

# Example medication row format:
# | Empagliflozin (SGLT2i) |  | 3–4 days pre-op | Eating/drinking normally, hemodynamically stable, no AKI | Endocrinology / ADA peri-op consensus |

# Specialty Rationale Template
SPECIALTY_RATIONALE_TEMPLATE = """
## 🩺 Specialty Rationale

### Cardiology Assessment
{cardiology_summary}

### Nephrology Assessment
{nephrology_summary}

### Endocrinology Assessment
{endocrinology_summary}

### Areas of Agreement
{agreements}

### Conflict Resolution
{conflicts_resolved}

---
*Reply: **A** for peri-op medication table | **C** for citations | **Back** to return to snapshot*
"""

# Citations Template
CITATIONS_TEMPLATE = """
## 📚 Clinical Guidelines & Citations

### Cardiology
{cardiology_citations}

### Nephrology
{nephrology_citations}

### Endocrinology
{endocrinology_citations}

---
*Reply: **A** for peri-op medication table | **B** for specialty rationale | **Back** to return to snapshot*
"""

# Standard medication table for common peri-op situations
STANDARD_PERIOP_MEDICATIONS = {
    "sglt2i": {
        "name": "SGLT2 inhibitor (e.g., empagliflozin, dapagliflozin)",
        "continue": "",
        "hold": "3–4 days pre-op",
        "restart": "Eating/drinking normally, hemodynamically stable, no AKI",
        "owner": "Endocrinology / Anesthesia (ADA, peri-op safety consensus)"
    },
    "metformin": {
        "name": "Metformin",
        "continue": "",
        "hold": "Day of surgery (and 48h post-op)",
        "restart": "eGFR stable, no AKI, contrast risk resolved",
        "owner": "Endocrinology (ADA, peri-op safety)"
    },
    "acei_arb": {
        "name": "ACE inhibitor / ARB (e.g., lisinopril, losartan)",
        "continue": "",
        "hold": "24h pre-op",
        "restart": "Hemodynamically stable, euvolemic, K acceptable",
        "owner": "Nephrology / Anesthesia"
    },
    "beta_blocker": {
        "name": "Beta-blocker (e.g., carvedilol, metoprolol)",
        "continue": "✓",
        "hold": "",
        "restart": "Continue peri-op; avoid abrupt withdrawal",
        "owner": "Cardiology (peri-op cardiac risk management)"
    },
    "statin": {
        "name": "Statin (e.g., atorvastatin, rosuvastatin)",
        "continue": "✓",
        "hold": "",
        "restart": "Continue peri-op",
        "owner": "Cardiology"
    },
    "loop_diuretic": {
        "name": "Loop diuretic (e.g., furosemide)",
        "continue": "Conditional",
        "hold": "Day of surgery if hypovolemic or NPO",
        "restart": "Based on volume status and renal function",
        "owner": "Cardiology / Anesthesia"
    },
    "aspirin": {
        "name": "Aspirin",
        "continue": "Conditional",
        "hold": "Case-dependent",
        "restart": "Restart once surgical hemostasis secured",
        "owner": "Surgery (bleeding risk) + Cardiology (primary vs secondary prevention)"
    },
    "anticoagulant": {
        "name": "Anticoagulant (e.g., warfarin, apixaban)",
        "continue": "",
        "hold": "Per anticoagulation protocol (3–5 days pre-op)",
        "restart": "Based on bleeding risk and indication",
        "owner": "Hematology / Cardiology / Surgery"
    },
    "insulin": {
        "name": "Insulin (basal/bolus)",
        "continue": "Conditional",
        "hold": "Short-acting: hold on morning of surgery",
        "restart": "Resume with meals; adjust based on NPO status",
        "owner": "Endocrinology (ADA perioperative guidelines)"
    },
    "sulfonylurea": {
        "name": "Sulfonylurea (e.g., glipizide, glyburide)",
        "continue": "",
        "hold": "Day of surgery",
        "restart": "Resume with meals to avoid hypoglycemia",
        "owner": "Endocrinology (ADA perioperative guidelines)"
    },
    "glp1ra": {
        "name": "GLP-1 RA (e.g., semaglutide, liraglutide)",
        "continue": "Conditional",
        "hold": "Weekly formulations: hold 1 week pre-op (aspiration risk)",
        "restart": "After return of normal GI function",
        "owner": "Endocrinology / Anesthesia (ASA 2023 guidance)"
    },
}


def format_medication_table(medications: list[str]) -> str:
    """Format a list of medication keys into a markdown table.
    
    Args:
        medications: List of medication keys from STANDARD_PERIOP_MEDICATIONS
        
    Returns:
        Formatted markdown table rows
    """
    rows = []
    for med_key in medications:
        if med_key in STANDARD_PERIOP_MEDICATIONS:
            med = STANDARD_PERIOP_MEDICATIONS[med_key]
            row = f"| {med['name']} | {med['continue']} | {med['hold']} | {med['restart']} | {med['owner']} |"
            rows.append(row)
    return "\n".join(rows)


def generate_consultation_snapshot(
    one_line_problem: str,
    key_facts: list[str],
    key_risks: list[str],
    decisions_needed: str,
    next_steps: list[dict]
) -> str:
    """Generate a formatted Consultation Snapshot.
    
    Args:
        one_line_problem: Single sentence summarizing the case
        key_facts: List of 5 key clinical facts
        key_risks: List of 5 key risks
        decisions_needed: Yes/No with brief explanation
        next_steps: List of dicts with 'action', 'owner', 'timing' keys
        
    Returns:
        Formatted Consultation Snapshot string
    """
    facts_formatted = "\n".join(f"  {i+1}. {fact}" for i, fact in enumerate(key_facts[:5]))
    risks_formatted = "\n".join(f"  {i+1}. {risk}" for i, risk in enumerate(key_risks[:5]))
    
    steps_formatted = "\n".join(
        f"  • **{step.get('action', '')}** — {step.get('owner', '')} ({step.get('timing', '')})"
        for step in next_steps
    )
    
    return CONSULTATION_SNAPSHOT_TEMPLATE.format(
        one_line_problem=one_line_problem,
        key_facts=facts_formatted,
        key_risks=risks_formatted,
        decisions_needed=decisions_needed,
        next_steps=steps_formatted
    )


# Backwards compatibility alias
generate_board_snapshot = generate_consultation_snapshot
BOARD_SNAPSHOT_TEMPLATE = CONSULTATION_SNAPSHOT_TEMPLATE

