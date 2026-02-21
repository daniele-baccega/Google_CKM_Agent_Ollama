"""
Baseline Single-Agent for CKM Study Control Group.
Runs the same Qwen2.5:14b model but with a generic, single-shot prompt.
"""
from litellm import completion

# Initialize the same local model used in the MAS
MODEL_NAME = "ollama_chat/ministral-3:14b"

GENERIC_SYSTEM_PROMPT = """
You are an expert multidisciplinary physician specializing in Cardio-Kidney-Metabolic (CKM) Syndrome. 
You are well-versed in ESC, AHA, KDIGO, and ADA guidelines.

Review the patient case provided below and generate a clinical consultation summary. 
Include:
1. Key Diagnoses
2. Immediate Risks
3. Medication Recommendations (Start/Stop/Adjust)
4. Peri-operative instructions (if applicable)

Be concise and professional.
"""

def run_baseline(case_text):
    response = completion(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": GENERIC_SYSTEM_PROMPT + f"\n\nPATIENT CASE:\n{case_text}"
        }],
        temperature=0,
        seed=0,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("--- CKM Baseline Agent (Control) ---")
    print("Paste your case below (Press Ctrl+D or Ctrl+Z on new line to submit):")
    
    # Simple multi-line input loop
    import sys
    case_input = sys.stdin.read()
    
    if case_input:
        print("\nGenerating Standard LLM Response...\n")
        output = run_baseline(case_input)
        print("="*60)
        print(output)
        print("="*60)