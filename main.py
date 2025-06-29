import os
import openai 
import string
import pandas as pd
import re
from openai import OpenAI
from tqdm import tqdm

def clean_sentence(text: str) -> str:
    # Remove punctuation
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    # Remove newlines and carriage returns
    text_no_newlines = text_no_punct.replace('\n', ' ').replace('\r', ' ').replace("**", "")
    # Convert to lowercase and strip extra whitespace
    return text_no_newlines.lower().strip()

def extract_description(prompt):
    # Replace newlines with spaces
    prompt_cleaned = prompt.replace('\n', ' ')

    # Find the first full stop and extract everything after it
    match = re.search(r'\.(.*)', prompt_cleaned)
    return match.group(1).strip() if match else None

def clean_summary(summary):
    sentences = summary.split('.')
        # Remove any sentence that contains 'I am a nurse' (case insensitive)
    filtered = [s for s in sentences if 'i am a nurse' not in s.lower()]
    # Join the filtered sentences back together
    cleaned_summary = '. '.join(filtered).strip()
    return cleaned_summary

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)
test = pd.read_csv('Data/test_raw.csv')

# Optimized prompt with enhanced structure and clinical tone
optimized_prompt_template = """
You are a Kenyan clinical officer. Given a medical scenario written by a nurse, your task is to:

1. Summarize the patient case using professional clinical language and management steps.
3. Ensure the format and style matches the following examples exactly.

### Example Outputs:

Summary:
A 4-year-old with 5% superficial burns. No other injuries.

Immediate Management:
 * Paracetamol analgesics to ensure child has minimal or no pain
 * Cleaning and dressing of wound with silver sulfadiazine
 * Topical prophylactic considered in this case

Follow-up Care:
 * Good nutrition – high-protein diet

**********************************

Summary:
6-year-old with vomiting and abdominal pain. Known diabetic on insulin but non-adherent due to lack of funds. Confused, Kussmaul breathing, fruity breath, dry tongue, blurred vision.Temp (normal), Pulse ↑120, RR ↑48, SpO₂ ↓90%.

Diagnosis:
Diabetic Ketoacidosis (DKA) due to insulin omission in a type 1 diabetic patient.

Immediate Management:
 * Insert IV line and administer normal saline bolus
 * Continuous insulin infusion (0.1 U/kg/hr)
 * Monitor blood glucose
 * Add potassium to IV fluids if hypokalemic
 * Treat underlying infection
 * Monitor ketones, pH, and bicarbonate
 
 Investigations:
 * Urinalysis
 * Blood gas analysis
 * Random blood sugar (RBS)
 * HbA1c
 * UECs
 * CBC
**********************************

### INPUT:
{{question}}
### OUTPUT:
"""

for idx , row in tqdm(test.iterrows(), total=len(test), desc="Processing test rows"):
    question = row['Prompt']
    cleaned_prompt =  "summary "+ " ".join(question.split('.')[1:]).replace('\n\n', ' ').replace('\n', ' ')
    formatted_prompt = optimized_prompt_template.replace("{{question}}", question)
    response = client.chat.completions.create(
        model = "qwen2.5:0.5b",
        seed= 42,
        temperature=0,
        max_tokens=256,

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant!'},
            {'role': 'user', 'content': formatted_prompt}
        ]
    )
    response_text = f"{cleaned_prompt} {response.choices[0].message.content.strip()}"
    processed_response = response_text.replace('\n\n', ' ').replace('\n', ' ').replace('**', '').replace("####", "").strip().lower()
    test.at[idx, 'Clinician'] = f"{processed_response}"

test[['Master_Index', 'Clinician']].to_csv("Data/optimizedv2_test_raw_with_prompt.csv", index=False)
print(test['Clinician'][0])
    






