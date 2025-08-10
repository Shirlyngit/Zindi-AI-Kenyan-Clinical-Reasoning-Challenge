EXAMPLE_FEW_SHOT = """
You are a Kenyan clinical officer. Given a medical scenario written by a nurse, your task is to:

1. Summarize the patient case using professional clinical language.
2. Provide a structured response with diagnosis, investigations, management, and follow-up.
3. Ensure the format and style matches the following examples exactly.

### Example 1:
I am a nurse with 18 years of experience in General nursing working in a Sub-county Hospitals and Nursing Homes in Uasin Gishu county in Kenya. A 4-year-old child presents to the emergency department with second-degree burns on the forearm after accidentally touching a hot stove. The child was playing in the kitchen when they reached out to touch the stove. The burns cover about 5% of the total body surface area. The child is alert and crying, with redness, blisters, and swelling on the affected area...
Questions:
1. What is the immediate treatment protocol for second-degree burns in paediatric patients?
2. Should any tetanus prophylaxis be considered in this case?
3. What follow-up care should be recommended for burn healing?
----------------------------------------------------------------------------------------------------
Summary:
A 4-year-old with 5% superficial burns. No other injuries.

Immediate Management:
 * Paracetamol analgesics to ensure child has minimal or no pain
 * Cleaning and dressing of wound with silver sulfadiazine
 * Topical prophylactic considered in this case

Follow-up Care:
 * Good nutrition – high-protein diet

**********************************

### Example 2:
I am a nurse with 17 years of experience in General nursing working in a National Referral Hospitals in Uasin Gishu county in Kenya. A 6-year-old girl presented to the emergency department with complaints of vomiting and abdominal pains. Patient is known diabetic on insulin and does not take as scheduled because of lack of funds.
On assessment the girl is confused, having Kussmaul breathing and fruity-scented breath. Has dry tongue and reports blurred vision. Temp 37°C, Pulse 120 bpm, Resp 48 bpm (rapid and labored), SpO2 90% on room air.
Questions:
What is the diagnosis of the patient?
What is the most immediate management?
What health education will be given to the patient and family?
Which investigations will be ordered?
----------------------------------------------------------------------------------------------------
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
"""


def build_prompt(clinical_note: str) -> str:
    directive = (
        "Now summarize the following clinical nurse note. Output must be structured exactly as in examples "
        "(Summary, Diagnosis:, Immediate Management:, Investigations:, Follow-up Care:, Medications:). "
        "Medications should be a bullet list with name, dose (if known or 'tbd'), route, and frequency if available.\n\n"
    )
    return EXAMPLE_FEW_SHOT + "\n\n" + directive + "Clinical note:\n" + clinical_note + "\n\n----\nResponse:\n"
