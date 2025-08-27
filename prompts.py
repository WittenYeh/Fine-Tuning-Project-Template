# src/prompts.py

def prompt_base(text):
    return f"""
You are a medical expert tasked with inferring a patient's smoking status from the following medical record text.

For each case text, you need to analyze and determine its condition.
- Past Smoker: A patient who smoked before but quit at least one year ago.
- Current Smoker: A patient who has smoked within the past year.
- Smoker: A patient whose record mentions smoking but does not provide enough details to classify them as a Past or Current Smoker.
- Non-Smoker: A patient whose record explicitly states that they never smoked.
- Unknown: A record that does not mention smoking status.

Here is the transcript of the medical record:
{text}

Please respond with only one of the following categories (no additional explanation or details). No other options, just the five options I've provided:
- PAST SMOKER
- CURRENT SMOKER
- SMOKER
- NON-SMOKER
- UNKNOWN
"""

def prompt_example(text):
    return f"""
You are a medical expert trained to assess smoking status from clinical notes.

For each case text, you need to analyze and determine its condition.

- Past Smoker: A patient who smoked before but quit at least one year ago.
- Current Smoker: A patient who has smoked within the past year.
- Smoker: A patient whose record mentions smoking but does not provide enough details to classify them as a Past or Current Smoker.
- Non-Smoker: A patient whose record explicitly states that they never smoked.
- Unknown: A record that does not mention smoking status.

Here are five text examples and five corresponding states for your reference.
Examples:
1. Patient Lucas has a documented smoking history but has not smoked cigarettes for the past two years. His recent chest X-ray was clear, and he reports improved respiratory function.
Category: PAST SMOKER

2. Patient Emma is a smoker who currently smokes approximately one pack per day. She was advised to consider smoking cessation therapy during her last clinic visit six months ago.
Category: CURRENT SMOKER

3. Patient Oliver has a history of smoking, and lung scans revealed mild abnormalities. The specific timing of his smoking cessation was not documented clearly in the medical records.
Category: SMOKER

4. Patient Sophia is a 25-year-old who has never smoked. She presented with symptoms of gastritis and has no reported respiratory complaints.
Category: NON-SMOKER

5. Patient William presented to the emergency department with a sprained ankle sustained during sports activity. He was treated with ice packs and advised rest for two weeks.
Category: UNKNOWN

Here is the transcript of the medical record:
{text}

Please respond with only one of the following categories (no additional explanation or details):
- PAST SMOKER
- CURRENT SMOKER
- SMOKER
- NON-SMOKER
- UNKNOWN
"""

def prompt_chain(text):
    return f"""
You are a medical expert trained to assess smoking status from clinical notes.

For each case text, you need to analyze and determine its condition.

- Past Smoker: A patient who smoked before but quit at least one year ago.
- Current Smoker: A patient who has smoked within the past year.
- Smoker: A patient whose record mentions smoking but does not provide enough details to classify them as a Past or Current Smoker.
- Non-Smoker: A patient whose record explicitly states that they never smoked.
- Unknown: A record that does not mention smoking status.

Here are five text examples and five corresponding states for your reference.
1. Patient Lucas has a documented smoking history but has not smoked cigarettes for the past two years. His recent chest X-ray was clear, and he reports improved respiratory function.
Category: PAST SMOKER

2. Patient Emma is a smoker who currently smokes approximately one pack per day. She was advised to consider smoking cessation therapy during her last clinic visit six months ago.
Category: CURRENT SMOKER

3. Patient Oliver has a history of smoking, and lung scans revealed mild abnormalities. The specific timing of his smoking cessation was not documented clearly in the medical records.
Category: SMOKER

4. Patient Sophia is a 25-year-old who has never smoked. She presented with symptoms of gastritis and has no reported respiratory complaints.
Category: NON-SMOKER

5. Patient William presented to the emergency department with a sprained ankle sustained during sports activity. He was treated with ice packs and advised rest for two weeks.
Category: UNKNOWN

Follow this 3-step reasoning to classify the patient's smoking status:

1.  **Smoking-Related?** If not, output UNKNOWN. Otherwise, continue.
2.  **Smoker vs. Non-Smoker?** If a non-smoker, output NON-SMOKER. Otherwise, continue.
3.  **Categorize Smoker Type:**
    - **PAST SMOKER**: Quit over a year ago.
    - **CURRENT SMOKER**: Smoked within the last year or is labeled 'current'.
    - **SMOKER**: Smoking is mentioned but with an unclear timeline.
    
Here is the transcript of the medical record:
{text}

Now read the following clinical note and respond step by step.

Please respond with only one of the following categories (no additional explanation or details):
- PAST SMOKER
- CURRENT SMOKER
- SMOKER
- NON-SMOKER
- UNKNOWN
"""

PROMPT_DICT = {
    "base": prompt_base,
    "example": prompt_example,
    "chain": prompt_chain,
}