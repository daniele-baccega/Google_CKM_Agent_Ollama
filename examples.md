# Examples

This document contains example queries and use cases for the CKM Syndrome Multi-Specialist Consultation portal.

## New UX Flow Overview

The system now features a **streamlined intake experience** with two modes:

### Welcome Message

When starting a conversation, you'll see:

> **Welcome to the Cardio-Kidney-Metabolic (CKM) Syndrome Multi-Specialist Consultation portal.**
>
> I help clinicians prepare and synthesize complex CKM cases involving the interplay of heart failure, chronic kidney disease, and metabolic conditions (diabetes, obesity). Recommendations follow current guidelines from cardiology (ESC/AHA), nephrology (KDIGO), and endocrinology (ADA).
>
> **Choose your intake mode:**
>
> 1. **Guided intake** *(recommended)* — I'll ask 3–5 high-yield questions step by step
> 2. **Paste mode** — Paste the full case (free text or JSON) and I'll structure it
>
> Reply **1** or **2** to begin.

---

## Example 1: Guided Intake Flow

### Step 1: User selects guided intake

**User:** `1`

**System Response:**

> Great! Let's start with the essential information.
>
> 1. What is the **primary clinical question** today? (e.g., medication optimization, peri-operative clearance, new diagnosis workup, decompensation management)
> 2. Is this a **peri-operative consultation**? Reply Yes/No.

### Step 2: User provides initial info

**User:**

```text
Medication optimization for CKM syndrome patient with recent HF decompensation.
No, not peri-operative.
```

**System Response:**

> Thank you. Please provide **CKM essentials**:
>
> • **Cardiac**: Ejection fraction (EF%), recent echo findings, NYHA class, BNP/NT-proBNP
> • **Kidney**: eGFR or creatinine, CKD stage, proteinuria (UACR if known)
> • **Metabolic**: HbA1c, diabetes type, BMI if available

### Step 3: User provides CKM essentials

**User:**

```text
Cardiac: EF 35%, dilated LV, NYHA Class III, NT-proBNP 1200
Kidney: eGFR 42, CKD Stage 3b, UACR 180 mg/g
Metabolic: HbA1c 8.1%, T2DM, BMI 32
```

**System Response:**

> Please list **current medications** (especially):
>
> - SGLT2 inhibitors (e.g., empagliflozin, dapagliflozin)
> - GLP-1 receptor agonists (e.g., semaglutide, liraglutide)
> - Metformin
> - ACE inhibitors/ARBs/ARNIs
> - Beta-blockers
> - MRAs (spironolactone, eplerenone)
> - Diuretics
> - Anticoagulants/Antiplatelets
> - Statins

### Step 4: User provides medications

**User:**

```text
Metformin 1000mg BID
Lisinopril 20mg daily
Carvedilol 12.5mg BID
Furosemide 40mg daily
Atorvastatin 40mg daily
```

**System Response:**

> Any **additional concerns** for the specialist panel?
>
> Or reply **'Generate synthesis'** to proceed with the consultation.

### Step 5: User requests synthesis

**User:** `Generate synthesis`

---

## Example Consultation Snapshot Output

After the specialist panel completes its review, you'll receive a **Consultation Snapshot** (≤250 words):

---

## 📋 Consultation Snapshot

**A) One-Line Problem:**
68M with CKD 3b, HFrEF (EF 35%), T2DM presenting for medication optimization after recent HF decompensation

**B) 5 Key Facts:**

1. EF 35% with dilated LV (HFrEF, NYHA III)
2. eGFR 42 mL/min/1.73m² (CKD Stage 3b) with albuminuria (UACR 180)
3. HbA1c 8.1% with BMI 32 (suboptimal glycemic control)
4. Not on SGLT2i, GLP-1 RA, MRA, or ARNI despite indications
5. On ACEi (not ARNI), beta-blocker at subtherapeutic dose

**C) 5 Key Risks:**

1. HF progression without GDMT optimization
2. CKD progression with uncontrolled albuminuria
3. Hypoglycemia risk if adding SGLT2i without adjusting other agents
4. Hyperkalemia risk if adding MRA with current eGFR
5. Volume overload with current diuretic regimen

**D) Decisions Needed Today:**
Yes — GDMT optimization: add SGLT2i, consider ARNI switch, uptitrate beta-blocker

**E) Next Steps:**

- **Add dapagliflozin 10mg** — Cardiology/Nephrology (start immediately, covers HF + CKD + T2DM)
- **Switch lisinopril → sacubitril/valsartan** — Cardiology (initiate after 36h ACEi washout)
- **Uptitrate carvedilol to 25mg BID** — Cardiology (every 2 weeks as tolerated)
- **Consider adding low-dose spironolactone** — Cardiology (monitor K and eGFR)
- **Reassess metformin dose** — Endocrinology (may reduce if adding SGLT2i)
- **Recheck labs in 1-2 weeks** — Primary care (K, creatinine, eGFR)

---

*Reply: **A** for peri-op medication stoplight table | **B** for specialty rationale | **C** for citations*

---

## Example 2: Peri-operative Consultation

### Step 1: User selects guided intake

**User:** `1`

**System Response:**

> Great! Let's start with the essential information.
>
> 1. What is the **primary clinical question** today?
> 2. Is this a **peri-operative consultation**? Reply Yes/No.

### Step 2: User indicates peri-op case

**User:**

```text
Peri-operative clearance for elective laparoscopic cholecystectomy.
Yes, peri-operative.
```

**System Response:**

> Please provide **procedure details**:
>
> - Type of surgery/procedure
> - Urgency (elective/urgent/emergent)
> - Expected duration and blood loss risk
> - Contrast use planned?

### Step 3: User provides procedure details

**User:**

```text
Laparoscopic cholecystectomy
Elective
~1 hour, minimal blood loss expected
No contrast
```

**System Response:**

> Now I need the **CKM essentials**:
>
> • **Cardiac**: Ejection fraction (EF%), recent echo findings, NYHA class, BNP/NT-proBNP
> • **Kidney**: eGFR or creatinine, CKD stage, proteinuria (UACR if known)
> • **Metabolic**: HbA1c, diabetes type, BMI if available

---

## Example 3: Paste Mode Flow

### Step 1: User selects paste mode

**User:** `2`

**System Response:**

> Please paste your case (free text or JSON format). I'll structure it for the specialist panel.

### Step 2: User pastes full case

**User:**

```text
72-year-old female patient

Demographics:
- Age: 72 years
- Sex: Female
- Weight: 85 kg
- Height: 165 cm

Medical History:
- Type 2 diabetes, diagnosed 2015
- Essential hypertension
- CKD Stage 3b
- HFpEF (last echo: EF 58%)

Vital Signs:
- Blood pressure: 138/82 mmHg
- Heart rate: 82 bpm

Laboratory Results:
- HbA1c: 7.8%
- eGFR: 45 mL/min/1.73m²
- Creatinine: 1.6 mg/dL
- Glucose: 165 mg/dL
- NT-proBNP: 320 pg/mL
- UACR: 120 mg/g

Current Medications:
- Empagliflozin 10mg daily
- Metformin 500mg BID
- Glipizide 5mg daily
- Losartan 50mg daily
- Carvedilol 6.25mg BID
- Atorvastatin 20mg daily

Chief Complaint:
Medication optimization - is current regimen adequate for CKM syndrome?
```

**System Response:**

> **Case Summary Extracted:**
>
> | Field | Value |
> |-------|-------|
> | Primary Question | Medication optimization for CKM syndrome |
> | Peri-operative | No |
> | Cardiac | EF 58% (HFpEF), NT-proBNP 320 |
> | Kidney | eGFR 45, CKD Stage 3b, UACR 120 mg/g |
> | Metabolic | HbA1c 7.8%, T2DM, BMI ~31 |
> | Medications | Empagliflozin, Metformin, Glipizide, Losartan, Carvedilol, Atorvastatin |
>
> Is this correct? Reply **'Confirm'** to proceed or provide corrections.

### Step 3: User confirms

**User:** `Confirm`

The system then generates the Consultation Snapshot.

---

## Expansion Examples

### Reply A — Peri-op Medication Stoplight Table

**User:** `A`

**System Response:**

## 💊 Peri-op Medication Stoplight Table

| Medication             | Continue |             Hold             | Restart Criteria                                         | Owner / Guideline                |
| ---------------------- | :------: | :--------------------------: | -------------------------------------------------------- | -------------------------------- |
| Empagliflozin (SGLT2i) |          |       3–4 days pre-op        | Eating/drinking normally, hemodynamically stable, no AKI | Endocrinology / Anesthesia (ADA) |
| Metformin              |          | Day of surgery (48h post-op) | eGFR stable, no AKI, contrast risk resolved              | Endocrinology (ADA)              |
| Glipizide (SU)         |          |        Day of surgery        | Resume with meals to avoid hypoglycemia                  | Endocrinology (ADA)              |
| Losartan (ARB)         |          |          24h pre-op          | Hemodynamically stable, euvolemic, K acceptable          | Nephrology / Anesthesia          |
| Carvedilol (β-blocker) |    ✓     |                              | Continue peri-op; avoid abrupt withdrawal                | Cardiology                       |
| Atorvastatin           |    ✓     |                              | Continue peri-op                                         | Cardiology                       |

---

*Reply: **B** for specialty rationale | **C** for citations | **Back** to return to snapshot*

---

### Reply B — Specialty Rationale

**User:** `B`

**System Response:**

## 🩺 Specialty Rationale

### Cardiology Assessment

- HFpEF with EF 58%, well-compensated (NT-proBNP 320)
- Continue beta-blocker for rate control and cardioprotection
- Consider uptitration if heart rate and BP allow

### Nephrology Assessment

- CKD Stage 3b (eGFR 45) with moderate albuminuria
- Already on SGLT2i and ARB — optimized for kidney protection
- Monitor for hyperkalemia if considering MRA

### Endocrinology Assessment

- T2DM with near-target HbA1c 7.8%
- On SGLT2i (excellent for cardiorenal protection)
- Consider discontinuing sulfonylurea (hypoglycemia risk) and adding GLP-1 RA for additional cardiorenal benefit

### Areas of Agreement

- All specialists agree current SGLT2i is appropriate
- Consensus on continuing beta-blocker and statin
- Agreement that GLP-1 RA would provide additional benefit

### Conflict Resolution

- None — unanimous agreement on management direction

---

*Reply: **A** for peri-op medication table | **C** for citations | **Back** to return to snapshot*

---

### Reply C — Citations

**User:** `C`

**System Response:**

## 📚 Clinical Guidelines & Citations

### Cardiology

- ESC 2023 Heart Failure Guidelines
- AHA 2024 Heart Failure Guidelines
- 2014 ACC/AHA Perioperative Cardiovascular Evaluation Guidelines

### Nephrology

- KDIGO 2024 Clinical Practice Guideline for CKD
- KDIGO 2012 Clinical Practice Guideline for AKI
- CREDENCE and DAPA-CKD trial data (SGLT2i in CKD)

### Endocrinology

- ADA 2024 Standards of Care in Diabetes
- STEP and SUSTAIN trial data (GLP-1 RA cardiorenal outcomes)
- Consensus on SGLT2i as first-line for T2DM with CKD or HF

---

*Reply: **A** for peri-op medication table | **B** for specialty rationale | **Back** to return to snapshot*

---

## Usage Notes

- The system uses Ollama `qwen2.5:14b` model by default
- **Guided intake** is recommended for most cases
- **Paste mode** is faster if you have a pre-formatted case
- **Consultation Snapshot** is always ≤250 words
- Details are hidden behind A, B, C expansion codes
- Reply **Back** at any time to return to the Consultation Snapshot