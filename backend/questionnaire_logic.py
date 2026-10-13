QUESTIONS = [
    {'key': 'itching', 'text': 'Do you have itching on the affected area?'},
    {'key': 'redness', 'text': 'Is there redness or inflammation?'},
    {'key': 'dry_skin', 'text': 'Is the skin dry?'},
    {'key': 'flaky_skin', 'text': 'Is the skin flaky or peeling?'},
    {'key': 'ring_shaped_patches', 'text': 'Are there ring-shaped patches?'},
    {'key': 'circular_patches', 'text': 'Are there circular patches?'},
    {'key': 'affects_nails', 'text': 'Does it affect your nails (discoloration or thickening)?'},
    {'key': 'oily_skin_scalp', 'text': 'Is the affected area oily (face/scalp especially)?'},
    {'key': 'pimples', 'text': 'Are there pimples present?'},
    {'key': 'bumps', 'text': 'Are there bumps present?'},
    {'key': 'silvery_scales', 'text': 'Are there silvery-white scales on the skin?'},
    {'key': 'long_duration', 'text': 'Has this been present for more than 1 month?'},
    {'key': 'affects_face_mostly', 'text': 'Is it mostly on your face?'},
]

DISEASE_SYMPTOM_MAP = {
    'Acne and Rosacea Photos': ['pimples', 'bumps', 'redness', 'affects_face_mostly'],
    'Eczema Photos': ['itching', 'dry_skin', 'flaky_skin', 'redness', 'long_duration'],
    'Nail Fungus and other Nail Disease': ['affects_nails', 'long_duration'],
    'Psoriasis pictures Lichen Planus and related diseases': ['silvery_scales', 'dry_skin', 'flaky_skin', 'long_duration', 'itching'],
    'Seborrheic Keratoses and other Benign Tumors': ['oily_skin_scalp', 'long_duration'],
    'Tinea Ringworm Candidiasis and other Fungal Infections': ['ring_shaped_patches', 'circular_patches', 'itching'],
}

def get_diagnosis(answers):
    scores = {}
    for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
        score = sum(1 for s in symptoms if answers.get(s) is True)
        scores[disease] = score

    sorted_diseases = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_disease, top_score = sorted_diseases[0]

    if top_score == 0:
        return {
            'diagnosis': 'Unable to determine from symptoms — please try the image upload or consult a dermatologist.',
            'confidence': 0
        }

    total_symptoms = len(DISEASE_SYMPTOM_MAP[top_disease])
    confidence = round((top_score / total_symptoms) * 100, 2)

    return {
        'diagnosis': top_disease,
        'confidence': confidence
    }