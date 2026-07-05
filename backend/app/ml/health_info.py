"""Health information database for arrhythmia types.

Provides patient-friendly explanations for each arrhythmia class,
including what it is, causes, whether it's dangerous, when to see
a doctor, and lifestyle tips.
"""

HEALTH_INFO = {
    "N": {
        "name": "Normal Sinus Rhythm",
        "icon": "heart",
        "color": "emerald",
        "what_is_it": (
            "Normal sinus rhythm is the healthy, regular heartbeat pattern "
            "that originates from the sinoatrial (SA) node, the heart's "
            "natural pacemaker. The heart beats at a rate of 60-100 beats "
            "per minute with consistent intervals between beats."
        ),
        "causes": [
            "This is the normal, healthy state of the heart",
            "No underlying cardiac condition",
        ],
        "is_dangerous": (
            "Normal sinus rhythm is not dangerous. It indicates that the "
            "heart's electrical system is functioning properly."
        ),
        "when_to_see_doctor": [
            "If you experience symptoms despite normal rhythm (chest pain, dizziness)",
            "For regular check-ups if you have risk factors for heart disease",
        ],
        "lifestyle_tips": [
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains",
            "Exercise regularly (150 minutes of moderate activity per week)",
            "Manage stress through relaxation techniques",
            "Get adequate sleep (7-9 hours per night)",
            "Avoid smoking and limit alcohol consumption",
        ],
    },
    "PVC": {
        "name": "Premature Ventricular Contraction",
        "icon": "alert-triangle",
        "color": "amber",
        "what_is_it": (
            "A Premature Ventricular Contraction (PVC) is an early heartbeat "
            "that originates in the ventricles (the heart's lower chambers) "
            "instead of the SA node. The beat comes earlier than expected "
            "and is often followed by a brief pause, which can feel like a "
            "'skipped beat' or a 'thud' in the chest."
        ),
        "causes": [
            "Stress, anxiety, or strong emotions",
            "Caffeine (coffee, tea, energy drinks, chocolate)",
            "Alcohol consumption",
            "Electrolyte imbalances (low potassium or magnesium)",
            "Lack of sleep or fatigue",
            "Certain medications (cold medicines, asthma inhalers)",
            "Underlying heart disease or previous heart attack",
            "Thyroid disorders",
        ],
        "is_dangerous": (
            "Occasional PVCs (a few per day) are very common and usually "
            "harmless, even in healthy people. They occur in up to 75% of "
            "the general population. However, frequent PVCs (more than "
            "10,000 per day) or PVCs occurring in patterns (couplets, "
            "bigeminy) in people with structural heart disease may "
            "increase the risk of more serious arrhythmias."
        ),
        "when_to_see_doctor": [
            "PVCs occur frequently or are increasing in frequency",
            "You feel dizzy, lightheaded, or faint",
            "You experience chest pain or shortness of breath",
            "You have a history of heart disease or heart attack",
            "PVCs interfere with your daily activities or sleep",
        ],
        "lifestyle_tips": [
            "Reduce or eliminate caffeine intake",
            "Limit alcohol consumption",
            "Ensure adequate sleep and manage fatigue",
            "Manage stress through meditation, yoga, or deep breathing",
            "Maintain electrolyte balance (eat potassium-rich foods like bananas)",
            "Stay hydrated",
            "Avoid stimulants in cold medications if sensitive",
        ],
    },
    "PAC": {
        "name": "Premature Atrial Contraction",
        "icon": "alert-triangle",
        "color": "amber",
        "what_is_it": (
            "A Premature Atrial Contraction (PAC) is an early heartbeat that "
            "originates in the atria (the heart's upper chambers) instead of "
            "the SA node. Like PVCs, PACs can feel like a skipped beat or "
            "a brief fluttering sensation. PACs are generally less "
            "concerning than PVCs because they originate in the atria."
        ),
        "causes": [
            "Stress and anxiety",
            "Caffeine and stimulants",
            "Alcohol, especially in excess",
            "Fatigue and sleep deprivation",
            "Electrolyte imbalances",
            "Thyroid disorders (hyperthyroidism)",
            "Atrial enlargement or stretching",
        ],
        "is_dangerous": (
            "PACs are generally benign and very common, especially with "
            "aging. Most people experience occasional PACs without knowing "
            "it. In rare cases, very frequent PACs may be a precursor to "
            "atrial fibrillation. If you have frequent PACs, monitoring "
            "by a cardiologist is recommended."
        ),
        "when_to_see_doctor": [
            "PACs are very frequent or increasing",
            "You experience palpitations with dizziness or chest discomfort",
            "You have a family history of atrial fibrillation",
            "Symptoms interfere with daily life",
        ],
        "lifestyle_tips": [
            "Reduce caffeine and stimulant intake",
            "Limit alcohol consumption",
            "Maintain a regular sleep schedule",
            "Practice stress management techniques",
            "Stay well-hydrated",
            "Treat underlying thyroid conditions if present",
        ],
    },
    "LBBB": {
        "name": "Left Bundle Branch Block",
        "icon": "alert-circle",
        "color": "rose",
        "what_is_it": (
            "Left Bundle Branch Block (LBBB) is a condition where electrical "
            "signals to the left ventricle of the heart are delayed. This "
            "causes the left ventricle to contract later than the right "
            "ventricle, resulting in an abnormal ECG pattern with a wide "
            "QRS complex. LBBB affects the timing of the heartbeat but not "
            "necessarily the heart's ability to pump blood."
        ),
        "causes": [
            "Coronary artery disease",
            "High blood pressure (hypertension)",
            "Cardiomyopathy (disease of the heart muscle)",
            "Aortic valve disease",
            "Previous heart attack",
            "Ageing of the heart's conduction system",
            "Sometimes no underlying cause is found (idiopathic)",
        ],
        "is_dangerous": (
            "LBBB itself is not life-threatening, but it often indicates "
            "underlying heart disease. New-onset LBBB can sometimes signal "
            "an acute heart attack and requires immediate medical "
            "evaluation. Chronic LBBB (present for a long time) is usually "
            "monitored but may require treatment of the underlying "
            "condition."
        ),
        "when_to_see_doctor": [
            "If LBBB is newly diagnosed (to rule out acute cardiac events)",
            "If you experience chest pain, shortness of breath, or fainting",
            "If you have known heart disease",
            "For regular cardiac monitoring if LBBB is chronic",
        ],
        "lifestyle_tips": [
            "Control blood pressure through diet, exercise, and medication",
            "Manage cholesterol levels",
            "Follow a heart-healthy diet (Mediterranean or DASH diet)",
            "Exercise regularly as approved by your cardiologist",
            "Avoid smoking",
            "Limit alcohol intake",
            "Monitor for any new or worsening symptoms",
        ],
    },
    "RBBB": {
        "name": "Right Bundle Branch Block",
        "icon": "alert-circle",
        "color": "rose",
        "what_is_it": (
            "Right Bundle Branch Block (RBBB) is a condition where electrical "
            "signals to the right ventricle are delayed, causing it to "
            "contract later than the left ventricle. This produces a "
            "characteristic ECG pattern with a wide QRS complex. RBBB is "
            "generally less concerning than LBBB and can occur in "
            "otherwise healthy individuals."
        ),
        "causes": [
            "Often occurs in healthy people with no heart disease",
            "Atrial septal defect (a hole in the heart)",
            "Pulmonary embolism",
            "Right ventricular hypertrophy (enlarged right ventricle)",
            "Coronary artery disease",
            "Cardiomyopathy",
            "Previous heart surgery",
        ],
        "is_dangerous": (
            "RBBB in the absence of other heart disease is usually benign "
            "and does not require treatment. However, new RBBB combined "
            "with symptoms like chest pain or shortness of breath should "
            "be evaluated by a cardiologist to rule out underlying "
            "conditions."
        ),
        "when_to_see_doctor": [
            "If RBBB is newly diagnosed with symptoms",
            "If you experience chest pain, shortness of breath, or fainting",
            "If you have a history of congenital heart disease",
            "For regular check-ups if RBBB is known",
        ],
        "lifestyle_tips": [
            "Maintain a heart-healthy lifestyle",
            "Exercise regularly (with cardiologist approval)",
            "Control blood pressure and cholesterol",
            "Avoid smoking",
            "Maintain a healthy weight",
            "Report any new symptoms to your doctor promptly",
        ],
    },
    "AFib": {
        "name": "Atrial Fibrillation",
        "icon": "zap",
        "color": "rose",
        "what_is_it": (
            "Atrial Fibrillation (AFib) is the most common type of serious "
            "arrhythmia. It occurs when the atria (upper chambers of the "
            "heart) quiver rapidly and irregularly instead of contracting "
            "effectively. This causes an irregular and often rapid "
            "heartbeat. AFib can be paroxysmal (comes and goes), "
            "persistent (lasts more than 7 days), or permanent."
        ),
        "causes": [
            "High blood pressure (the most common cause)",
            "Coronary artery disease",
            "Heart valve disease",
            "Heart failure",
            "Overactive thyroid (hyperthyroidism)",
            "Excessive alcohol consumption (especially binge drinking)",
            "Obesity",
            "Sleep apnea",
            "Age (risk increases significantly after 65)",
            "Previous heart surgery",
        ],
        "is_dangerous": (
            "AFib itself is not immediately life-threatening, but it "
            "significantly increases the risk of stroke (5x higher than "
            "normal) because blood can pool in the atria and form clots. "
            "AFib can also lead to heart failure if the heart rate is "
            "not controlled. Proper medical management (blood thinners, "
            "rate or rhythm control medications) is essential to reduce "
            "complications."
        ),
        "when_to_see_doctor": [
            "If you experience irregular, rapid, or pounding heartbeat",
            "If you feel shortness of breath, chest pain, or dizziness",
            "If you have risk factors (high BP, diabetes, age >65)",
            "URGENT: Seek emergency care if you experience stroke symptoms "
            "(face drooping, arm weakness, speech difficulty)",
        ],
        "lifestyle_tips": [
            "Control blood pressure through medication and lifestyle",
            "Maintain a healthy weight",
            "Limit alcohol intake (especially binge drinking)",
            "Treat sleep apnea if diagnosed",
            "Manage stress levels",
            "Exercise regularly as approved by your doctor",
            "Follow a heart-healthy, low-sodium diet",
            "Take prescribed blood thinners to prevent stroke",
            "Monitor your pulse regularly for irregularity",
        ],
    },
}


def get_health_info(arrhythmia_class: str) -> dict:
    """Get health information for a given arrhythmia class.

    Args:
        arrhythmia_class: One of "N", "PVC", "PAC", "LBBB", "RBBB", "AFib"

    Returns:
        Dictionary with health information, or a default message if the
        class is not found.
    """
    return HEALTH_INFO.get(arrhythmia_class, {
        "name": "Unknown",
        "icon": "help-circle",
        "color": "slate",
        "what_is_it": "The detected rhythm pattern could not be classified.",
        "causes": [],
        "is_dangerous": "Please consult a physician for proper diagnosis.",
        "when_to_see_doctor": ["Consult a physician for evaluation"],
        "lifestyle_tips": [],
    })


def list_all_arrhythmia_types() -> list:
    """List all arrhythmia types with their health info."""
    return [
        {"code": k, **v}
        for k, v in HEALTH_INFO.items()
    ]
