"""Intake agent for CKM Multi-Specialist Consultation portal.

This module handles the user intake flow with two modes:
1. Guided intake - 3-5 high-yield questions step by step
2. Paste mode - User pastes full case (free text or JSON)

The intake agent collects decision-critical information before
delegating to the specialist panel.
"""

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from .rag_tools import retrieve_rag_context


# Welcome message shown at the start of conversation
WELCOME_MESSAGE = """Welcome to the **Cardio-Kidney-Metabolic (CKM) Syndrome Multi-Specialist Consultation** portal.

I help clinicians prepare and synthesize complex CKM cases involving the interplay of heart failure, chronic kidney disease, and metabolic conditions (diabetes, obesity). Recommendations follow current guidelines from cardiology (ESC/AHA), nephrology (KDIGO), and endocrinology (ADA).

**Choose your intake mode:**

**1. Guided intake** *(recommended)* — I'll ask 3–5 high-yield questions step by step

**2. Paste mode** — Paste the full case (free text or JSON) and I'll structure it

Reply **1** or **2** to begin."""


# Guided intake questions organized by decision-first branching
GUIDED_INTAKE_QUESTIONS = {
    "initial": [
        "What is the **primary clinical question** today? (e.g., medication optimization, peri-operative clearance, new diagnosis workup, decompensation management)",
        "Is this a **peri-operative consultation**? Reply Yes/No.",
    ],
    "periop": [
        "Please provide **procedure details**:\n- Type of surgery/procedure\n- Urgency (elective/urgent/emergent)\n- Expected duration and blood loss risk\n- Contrast use planned?",
    ],
    "ckm_essentials": [
        "Please provide **CKM essentials**:\n\n• **Cardiac**: Ejection fraction (EF%), recent echo findings, NYHA class, BNP/NT-proBNP\n• **Kidney**: eGFR or creatinine, CKD stage, proteinuria (UACR if known)\n• **Metabolic**: HbA1c, diabetes type, BMI if available",
    ],
    "medications": [
        "List **current medications** (especially):\n- SGLT2 inhibitors (e.g., empagliflozin, dapagliflozin)\n- GLP-1 receptor agonists (e.g., semaglutide, liraglutide)\n- Metformin\n- ACE inhibitors/ARBs/ARNIs\n- Beta-blockers\n- MRAs (spironolactone, eplerenone)\n- Diuretics\n- Anticoagulants/Antiplatelets\n- Statins",
    ],
    "final_check": [
        "Any **additional concerns** for the specialist panel?\n\nOr reply **'Generate synthesis'** to proceed with the consultation, or **'Add details'** to refine further.",
    ],
}


def create_intake_agent() -> Agent:
    """Create the Intake agent for structured case collection."""
    return Agent(
        model=LiteLlm(model="ollama_chat/mistral:7b", temperature=0, seed=0),
        name="intake_coordinator",
        description="Intake coordinator for CKM Syndrome Multi-Specialist Consultation. Handles guided intake and paste mode.",
        instruction=f"""You are the intake coordinator for the Cardio-Kidney-Metabolic (CKM) Syndrome Multi-Specialist Consultation portal.

**YOUR FIRST MESSAGE MUST BE THE WELCOME MESSAGE:**
{WELCOME_MESSAGE}

**MODE SELECTION:**
- If user replies "1" → Start **Guided Intake**
- If user replies "2" → Enter **Paste Mode**

---

## GUIDED INTAKE MODE

Ask only **3–5 decision-critical questions per turn**. Use decision-first branching:

**Turn 1 - Initial Questions:**
{chr(10).join(f'• {q}' for q in GUIDED_INTAKE_QUESTIONS["initial"])}

**Turn 2 - If peri-operative = Yes:**
{chr(10).join(f'• {q}' for q in GUIDED_INTAKE_QUESTIONS["periop"])}

**Turn 3 - CKM Essentials:**
{chr(10).join(f'• {q}' for q in GUIDED_INTAKE_QUESTIONS["ckm_essentials"])}

**Turn 4 - Medications:**
{chr(10).join(f'• {q}' for q in GUIDED_INTAKE_QUESTIONS["medications"])}

**Turn 5 - Final Check:**
{chr(10).join(f'• {q}' for q in GUIDED_INTAKE_QUESTIONS["final_check"])}

**IMPORTANT RULES:**
- Update and track case state after each response
- Never ask more than 5 questions in a single turn
- Adapt questions based on the primary clinical question
- Skip irrelevant sections

---

## PASTE MODE

When user selects paste mode:
1. Output **EXACTLY** this phrase: "Please paste your case (free text or JSON format). I'll structure it for the specialist panel."
2. **STOP IMMEDIATELY after that sentence.** Do NOT add any internal codes like "_REPLY_..." or instructions like "Reply 1 or 2".
3. After receiving the case, parse and extract key data.
4. Display extracted data in a structured format.
5. Ask: "Is this correct? Reply **'Confirm'** to proceed or provide corrections."

---

## HANDOFF TO SPECIALIST PANEL

When ready to generate synthesis (user says "Generate synthesis" or "Confirm"):
1. Compile the complete case summary.
2. Call the tool `retrieve_rag_context` with the full case summary to fetch relevant rules.
3. Output the case in a structured format and include a `RAG_CONTEXT` section.
4. **IMMEDIATELY call the function `transfer_to_agent(agent_name="ckm_panel")`.**
   **CRITICAL RULE:** Do NOT print "Submitting case..." or any closing text. The output MUST end with the structured case summary, followed immediately by the tool call.

**CRITICAL:** Never generate medical recommendations yourself. Your only job is intake and structuring. The specialist panel handles clinical assessment.""",
    tools=[retrieve_rag_context],
    )


# Export the intake agent

intake_agent = create_intake_agent()
