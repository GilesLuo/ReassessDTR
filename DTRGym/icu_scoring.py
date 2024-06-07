# from numba import njit, jit

def get_APACHE2_score(abp, temperature, heart_bpm, respiratory_rate,
                      FiO2, PaO2, PaCO2,  # oxygenation input
                      ph, sodium, potassium,
                      hematocrit, wbc, age, scd, aki, creatinine, gcs):
    """

    :param abp: mean blood pressure
    :param temperature: temperature
    :param heart_bpm: heart rate
    :param respiratory_rate: respiratory rate
    :param oxygenation:
            FiO2: fraction inspired oxygen
            PaO2: partial pressure of oxygen, use PO2 for approximation: partial pressure of oxygen
            PaCO2: partial pressure of carbon dioxide
            R=0.8, Patm=760, PH20=47
            https://www.nursingcenter.com/ncblog/august-2020/calculate-a-a-gradient
            A-a oxygen gradient = [(FiO2 x [Patm - PH2O]) - (PaCO2 ÷ R)] - PaO2
                                = [(FiO2 x [760 - 47]) - (PaCO2 ÷ 0.8)] - PaO2
            For FiO2 >0.5 record AaDO2
            For FiO2 <0.5 record only PaO2 (arterial partial pressure of O2):  oxygenation
    :param ph: ph
    :param sodium:  sodium
    :param potassium: potassium
    :param hematocrit: hematocrit
    :param wbc: white blood cell count
    :param age: age
    :param scd: Severe Chronic Disease
                This function accepts a total of 6 values for addressing Chronic Health Points and surgery status
                    # value="RO NSCD">elective postoperative, no SCD (Severe Chronic Diseases, see list)
                    # value="RO SCD">elective postoperative, SCD present
                    # value="NO NSCD">no surgery, no SCD
                    # value="NO SCD">no surgery, SCD present
                    # value="EO NSCD">emergency postoperative, no SCD
                    # value="EO SCD">emergency postoperative, SCD present
                    SCD list:
                        ALS (Lou Gehrig's Disease)
                        Alzheimer's Disease and other Dementias
                        Arthritis
                        Asthma
                        Cancer
                        Chronic Obstructive Pulmonary Disease (COPD)
                        Crohn's Disease, Ulcerative Colitis, Other Inflammatory Bowel Diseases, Irritable Bowel Syndrome
                        Cystic Fibrosis
                        Diabetes
                        Eating Disorders
                        Heart Disease
                        Obesity
                        Oral Health
                        Osteoporosis
                        Reflex Sympathetic Dystrophy (RSD) Syndrome
                        Tobacco Use and Related Conditions
    :param aki: Acute renal injury present? 1 for yes, 0 for no
    :param creatinine: creatinine
    :param gcs:glascow coma scale total
    """

    def calculate_single_scores(selector, value):
        # APACHEII Ranges
        score_table = {"abp"             : {"range": [0, 50, 70, 110, 130, 160, float("inf")],
                                            "score": [4, 2, 0, 2, 3, 4]},
                       "temperature"     : {"range": [0, 30, 32, 34, 36, 38.5, 39, 41, float("inf")],
                                            "score": [4, 3, 2, 1, 0, 1, 3, 4]},
                       "heart_bpm"       : {"range": [0, 40, 55, 70, 110, 140, 180, float("inf")],
                                            "score": [4, 3, 2, 0, 2, 3, 4]},
                       "respiratory_rate": {"range": [0, 6, 10, 12, 25, 35, 50, float("inf")],
                                            "score": [4, 2, 1, 0, 2, 3, 4]},
                       "oxygenation"     : {"range": [float('-inf'), 55, 61, 71, 200, 350, 500, float("inf")],
                                            "score": [4, 3, 1, 0, 2, 3, 4]},
                       "ph"              : {"range": [1, 7.15, 7.25, 7.33, 7.5, 7.6, 7.7, 14],
                                            "score": [4, 3, 2, 0, 1, 3, 4]},
                       "sodium"          : {"range": [0, 111, 120, 130, 150, 155, 160, 180, float("inf")],
                                            "score": [4, 3, 2, 0, 1, 2, 3, 4]},
                       "potassium"       : {"range": [0, 2.5, 3, 3.5, 5.5, 6, 7, float("inf")],
                                            "score": [4, 2, 1, 0, 1, 3, 4]},
                       "hematocrit"      : {"range": [0, 20, 30, 46, 50, 60, float("inf")],
                                            "score": [4, 2, 0, 1, 2, 4]},
                       "wbc"             : {"range": [0, 1, 3, 15, 20, 40, float("inf")],
                                            "score": [4, 2, 0, 1, 2, 4]},
                       "age"             : {"range": [0, 45, 55, 65, 75, 110],
                                            "score": [0, 2, 3, 5, 6]},
                       }

        score_dict = score_table[selector]
        assert len(score_dict['range']) == len(score_dict['score']) + 1
        for idx, upper_bound in enumerate(score_dict['range']):
            if value < upper_bound:
                return score_dict['score'][idx - 1]
        raise ValueError('{} not in range of {}'.format(selector, score_dict['range']))

    def calculate_apache2_physiology(abp, temperature, heart_bpm, respiratory_rate, oxygenation, ph, sodium, potassium,
                                     hematocrit, wbc, age):
        # print(args)
        score = 0

        for selector, value in {"abp"             : abp, "temperature": temperature, "heart_bpm": heart_bpm,
                                "respiratory_rate": respiratory_rate, "oxygenation": oxygenation,
                                "ph"              : ph, "sodium": sodium, "potassium": potassium,
                                "hematocrit"      : hematocrit, "wbc": wbc, "age": age}.items():
            result = calculate_single_scores(selector, value)
            if result is None:
                # print(args, selectors)
                raise ValueError(selector, value, 'not in value range.')
            score += result
        return score

    def chronic_health_score(value):
        if value == "RO NSCD":
            result = "0"
        elif value == "RO SCD":
            result = "2"
        elif value == "EO NSCD":
            result = "0"
        elif value == "EO SCD":
            result = "5"
        elif value == "NO NSCD":
            result = "0"
        elif value == "NO SCD":
            result = "5"
        return result

    def creatinine_score(aki, creatinine):
        score_table = {"range": [0, 0.6, 1.5, 2, 3.5, 30],
                       "score": [2, 0, 2, 3, 4]}
        assert len(score_table['range']) == len(score_table['score']) + 1
        magnif = 2 if aki == 0 else 1  # Double points for acute kidney injury

        for idx, upper_bound in enumerate(score_table['range']):
            if creatinine < upper_bound:
                return score_table['score'][idx - 1] * magnif

    creatinine = round(creatinine, 1)
    args = list(locals().values())
    if FiO2 > 0.5:
        oxygenation = ((FiO2 * (760 - 47)) - (PaCO2 / 0.8)) - PaO2
        # assert oxygenation >= 0, "FiO2:{}, PaCO2:{}, PaO2{}, age:{}".format(FiO2, PaCO2, PaO2, age)
    else:
        oxygenation = PaO2
    physiology = calculate_apache2_physiology(abp, temperature, heart_bpm, respiratory_rate, oxygenation, ph, sodium,
                                              potassium, hematocrit, wbc, age)
    health_points = int(chronic_health_score(scd))
    creatinine_points = creatinine_score(aki, creatinine)
    gcs_final = 15 - gcs
    sub_scores = {"APACHE2_physiology":int(physiology),  "APACHE2_health_points":int(health_points),
                  "APACHE2_creatinine_points":int(creatinine_points), "APACHE2_gcs":gcs_final}
    final_score = sum(sub_scores.values())
    assert 0 <= final_score <= 71, "wrong APACHE score"
    return final_score, sub_scores


def in_range(lowerlimit, upperlimit, value):
    assert lowerlimit <= upperlimit
    return lowerlimit <= value < upperlimit


def get_NEWS2_score(respiratory_rate, SpO2, on_vent, blood_pressure, heart_rate, is_CVPU, temperature, is_AHRF):
    # https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2

    score_table = {"respiratory_rate": {"range": [0, 9, 12, 21, 25, float("inf")],
                                        "score": [3, 1, 0, 2, 3]},
                   "SpO2_scale1"     : {"range": [0, 92, 94, 96, float("inf")],
                                        "score": [3, 2, 1, 0]},
                   "SpO2_scale2"     : {"on_vent": {"range": [93, 95, 97, 100.01],
                                                    "score": [4, 3, 2]},
                                        "on_air" : {"range": [0, 84, 86, 88, 93],
                                                    "score": [4, 3, 2, 0]}},
                   "on_vent"         : {"range": [0, 1, float("inf")],
                                        "score": [2, 0]},
                   "blood_pressure"  : {"range": [0, 90, 100, 110, 220, float('inf')],
                                        "score": [3, 2, 1, 0, 3]},
                   "heart_rate"      : {"range": [0, 40, 50, 90, 110, 130, float("inf")],
                                        "score": [3, 1, 0, 1, 2, 3]},
                   "is_CVPU"         : {"range": [0, 1, float("inf")],
                                        "score": [0, 3]},
                   "temperature"     : {"range": [0, 35, 36, 38, 39, float("inf")],
                                        "score": [3, 1, 0, 1, 2]}}

    def calculate_single_scores(selector, value):
        score_dict = score_table[selector]

        assert len(score_dict['range']) == len(score_dict['score']) + 1
        for idx, upper_bound in enumerate(score_dict['range']):
            if value < upper_bound:
                return score_dict['score'][idx - 1]
        raise ValueError('{} not in range of {}'.format(selector, score_dict['range']))

    scoring_dict = {"respiratory_rate": respiratory_rate,
                    "on_vent"         : on_vent, "blood_pressure": blood_pressure,
                    "heart_rate"      : heart_rate, "is_CVPU": is_CVPU, "temperature": temperature}
    total_score = 0

    # calculate spo2 score:
    if is_AHRF:
        str_on_vent = "on_vent" if on_vent else "on_air"
        score_dict = score_table["SpO2_scale2"][str_on_vent]
        assert len(score_dict['range']) == len(score_dict['score']) + 1
        if SpO2 >= score_dict['range'][-1]:
            raise ValueError("SpO2 exceeds 100%, input wrong?")

        for idx, upper_bound in enumerate(score_dict['range']):
            if SpO2 < upper_bound:
                total_score += score_dict['score'][idx - 1]
                break
    else:
        scoring_dict.update({"SpO2_scale1": SpO2})

    for selector, value in scoring_dict.items():
        total_score += calculate_single_scores(selector, value)
    return total_score


def get_SOFA_score(PaO2, gcs, blood_pressure,
                   Dopamine, dobutamine, epinephrine, norepinephrine,
                   bilirubin, platelets, creatinine):
    raise NotImplementedError


def get_qSOFA_score(respiratory_rate, blood_pressure, gcs):
    score = 0
    score += 1 if respiratory_rate >= 22 else 0
    score += 1 if blood_pressure >= 100 else 0
    score += 1 if gcs <= 14 else 0
    return score


def determine_HRF(PaCO2, pH):
    """
    Determines if the given PaCO2 and pH values indicate Hypercapnic Respiratory Failure (HRF).

    Parameters:
    - PaCO2 (float): arterial partial pressure of CO2 in mm Hg.
    - pH (float): arterial pH.

    Returns:
    - bool: True if HRF is indicated, False otherwise.
    """

    if PaCO2 > 45 and pH < 7.35:
        return True
    return False