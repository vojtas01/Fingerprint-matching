# This simple script is used to find out FAR and FRR values. The genuine_scores stores the scores that are
# result of comparing two identical fingerprints. The impostor_scores stores the scores that are result from
# comparing two different fingerprints. The calculate_far_frr function calculates the FAR and FRR values for a given threshold.
# You can read more about FAR and FRR here: https://recogtech.com/en/insights-en/far-and-frr-security-level-versus-ease-of-use/

genuine_scores = []

impostor_scores = []


def calculate_far_frr(genuine_scores, impostor_scores, threshold):
    false_accepts = sum(score >= threshold for score in impostor_scores)
    false_rejects = sum(score < threshold for score in genuine_scores)
    far = false_accepts / len(impostor_scores)
    frr = false_rejects / len(genuine_scores)
    return far, frr


print(calculate_far_frr(genuine_scores, impostor_scores, 0))
