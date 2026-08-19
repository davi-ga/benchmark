import numpy as np
import numba
import math
from _numba_helpers import _sanitize_for_numba, _deep_to_list
IDX_N = 0
IDX_BETA = 1
IDX_BETA_HOSPITAL = 2
IDX_SIGMA = 3
IDX_DELTA = 4
IDX_DELTA_HOSPITAL = 5
IDX_GAMMA = 6
IDX_KAPPA = 7
IDX_HOSP_RATE_GEN = 8
IDX_HOSP_RATE_ICU = 9
IDX_SYMPTOMS_TO_HOSP = 10
IDX_LOS_GEN = 11
IDX_LOS_ICU = 12
IDX_LOS_ICU_VENT = 13
IDX_FRAC_ICU_VENT = 14
IDX_MORT_FROM_ICU = 15
IDX_MORT_FROM_HOSP = 16
IDX_MORT_NO_ICU = 17
IDX_MORT_FROM_ICU_VENT = 18
IDX_MORT_NO_GEN = 19
IDX_BEDS_GEN = 20
IDX_BEDS_ICU = 21
N_PARAMS = 22

@numba.njit
def derivative(t):
    n = len(t)
    result = np.zeros(n)
    for i in range(1, n):
        result[i] = t[i] - t[i - 1]
    return result

@numba.njit
def seir_time_step(y, t, suppression_value, params):
    S = y[0]
    E = y[1]
    A = y[2]
    I = y[3]
    R = y[4]
    HNonICU = y[5]
    HICU = y[6]
    HICUVent = y[7]
    D = y[8]
    N = params[IDX_N]
    beta = params[IDX_BETA]
    beta_hospital = params[IDX_BETA_HOSPITAL]
    sigma = params[IDX_SIGMA]
    delta = params[IDX_DELTA]
    gamma = params[IDX_GAMMA]
    kappa = params[IDX_KAPPA]
    hosp_rate_gen = params[IDX_HOSP_RATE_GEN]
    hosp_rate_icu = params[IDX_HOSP_RATE_ICU]
    symptoms_to_hosp = params[IDX_SYMPTOMS_TO_HOSP]
    los_gen = params[IDX_LOS_GEN]
    los_icu = params[IDX_LOS_ICU]
    los_icu_vent = params[IDX_LOS_ICU_VENT]
    frac_icu_vent = params[IDX_FRAC_ICU_VENT]
    mort_from_icu = params[IDX_MORT_FROM_ICU]
    mort_from_hosp = params[IDX_MORT_FROM_HOSP]
    mort_no_icu = params[IDX_MORT_NO_ICU]
    mort_from_icu_vent = params[IDX_MORT_FROM_ICU_VENT]
    mort_no_gen = params[IDX_MORT_NO_GEN]
    beds_general = params[IDX_BEDS_GEN]
    beds_icu = params[IDX_BEDS_ICU]
    number_exposed = beta * suppression_value * S * (kappa * I + A) / N + beta_hospital * S * (HICU + HNonICU) / N
    dSdt = -number_exposed
    exposed_and_symptomatic = gamma * sigma * E
    exposed_and_asymptomatic = (1.0 - gamma) * sigma * E
    dEdt = number_exposed - exposed_and_symptomatic - exposed_and_asymptomatic
    asymptomatic_and_recovered = delta * A
    dAdt = exposed_and_asymptomatic - asymptomatic_and_recovered
    infected_and_recovered_no_hospital = delta * I
    infected_and_in_hospital_general = I * (hosp_rate_gen - hosp_rate_icu) / symptoms_to_hosp
    infected_and_in_hospital_icu = I * hosp_rate_icu / symptoms_to_hosp
    dIdt = exposed_and_symptomatic - infected_and_recovered_no_hospital - infected_and_in_hospital_general - infected_and_in_hospital_icu
    if HICU <= beds_icu:
        mortality_rate_ICU = mort_from_icu
    else:
        mortality_rate_ICU = mort_no_icu
    if HNonICU <= beds_general:
        mortality_rate_NonICU = mort_from_hosp
    else:
        mortality_rate_NonICU = mort_no_gen
    died_from_hosp = HNonICU * mortality_rate_NonICU / los_gen
    died_from_icu = HICU * (1.0 - frac_icu_vent) * mortality_rate_ICU / los_icu
    died_from_icu_vent = HICUVent * mort_from_icu_vent / los_icu_vent
    recovered_after_hospital_general = HNonICU * (1.0 - mortality_rate_NonICU) / los_gen
    recovered_from_icu_no_vent = HICU * (1.0 - mortality_rate_ICU) * (1.0 - frac_icu_vent) / los_icu
    max_mort_icu = mort_from_icu
    if mort_from_icu_vent > max_mort_icu:
        max_mort_icu = mort_from_icu_vent
    recovered_from_icu_vent = HICUVent * (1.0 - max_mort_icu) / los_icu_vent
    dHNonICU_dt = infected_and_in_hospital_general - recovered_after_hospital_general - died_from_hosp
    dHICU_dt = infected_and_in_hospital_icu - recovered_from_icu_no_vent - recovered_from_icu_vent - died_from_icu - died_from_icu_vent
    dHICUVent_dt = infected_and_in_hospital_icu * frac_icu_vent - HICUVent / los_icu_vent
    dTotalInfections_dt = exposed_and_symptomatic + exposed_and_asymptomatic
    dHAdmissions_general_dt = infected_and_in_hospital_general
    dHAdmissions_ICU_dt = infected_and_in_hospital_icu
    dRdt = asymptomatic_and_recovered + infected_and_recovered_no_hospital + recovered_after_hospital_general + recovered_from_icu_vent + recovered_from_icu_no_vent
    dDdt = died_from_icu + died_from_icu_vent + died_from_hosp
    dydt = np.zeros(12)
    dydt[0] = dSdt
    dydt[1] = dEdt
    dydt[2] = dAdt
    dydt[3] = dIdt
    dydt[4] = dRdt
    dydt[5] = dHNonICU_dt
    dydt[6] = dHICU_dt
    dydt[7] = dHICUVent_dt
    dydt[8] = dDdt
    dydt[9] = dHAdmissions_general_dt
    dydt[10] = dHAdmissions_ICU_dt
    dydt[11] = dTotalInfections_dt
    return dydt

@numba.njit
def _build_params_array_jit(N, R0, R0_hospital, sigma, delta, delta_hospital, gamma, kappa, hospitalization_rate_general, hospitalization_rate_icu, symptoms_to_hospital_days, hospitalization_length_of_stay_general, hospitalization_length_of_stay_icu, hospitalization_length_of_stay_icu_and_ventilator, fraction_icu_requiring_ventilator, mortality_rate_from_ICU, mortality_rate_from_hospital, mortality_rate_no_ICU_beds, mortality_rate_from_ICUVent, mortality_rate_no_general_beds, beds_general, beds_ICU):
    params = np.zeros(N_PARAMS)
    params[IDX_N] = N
    params[IDX_BETA] = R0 * delta
    params[IDX_BETA_HOSPITAL] = R0_hospital * delta_hospital
    params[IDX_SIGMA] = sigma
    params[IDX_DELTA] = delta
    params[IDX_DELTA_HOSPITAL] = delta_hospital
    params[IDX_GAMMA] = gamma
    params[IDX_KAPPA] = kappa
    params[IDX_HOSP_RATE_GEN] = hospitalization_rate_general
    params[IDX_HOSP_RATE_ICU] = hospitalization_rate_icu
    params[IDX_SYMPTOMS_TO_HOSP] = symptoms_to_hospital_days
    params[IDX_LOS_GEN] = hospitalization_length_of_stay_general
    params[IDX_LOS_ICU] = hospitalization_length_of_stay_icu
    params[IDX_LOS_ICU_VENT] = hospitalization_length_of_stay_icu_and_ventilator
    params[IDX_FRAC_ICU_VENT] = fraction_icu_requiring_ventilator
    params[IDX_MORT_FROM_ICU] = mortality_rate_from_ICU
    params[IDX_MORT_FROM_HOSP] = mortality_rate_from_hospital
    params[IDX_MORT_NO_ICU] = mortality_rate_no_ICU_beds
    params[IDX_MORT_FROM_ICU_VENT] = mortality_rate_from_ICUVent
    params[IDX_MORT_NO_GEN] = mortality_rate_no_general_beds
    params[IDX_BEDS_GEN] = beds_general
    params[IDX_BEDS_ICU] = beds_ICU
    return params

def build_params_array(N, R0, R0_hospital, sigma, delta, delta_hospital, gamma, kappa, hospitalization_rate_general, hospitalization_rate_icu, symptoms_to_hospital_days, hospitalization_length_of_stay_general, hospitalization_length_of_stay_icu, hospitalization_length_of_stay_icu_and_ventilator, fraction_icu_requiring_ventilator, mortality_rate_from_ICU, mortality_rate_from_hospital, mortality_rate_no_ICU_beds, mortality_rate_from_ICUVent, mortality_rate_no_general_beds, beds_general, beds_ICU):
    return _deep_to_list(_build_params_array_jit(_sanitize_for_numba(N), _sanitize_for_numba(R0), _sanitize_for_numba(R0_hospital), _sanitize_for_numba(sigma), _sanitize_for_numba(delta), _sanitize_for_numba(delta_hospital), _sanitize_for_numba(gamma), _sanitize_for_numba(kappa), _sanitize_for_numba(hospitalization_rate_general), _sanitize_for_numba(hospitalization_rate_icu), _sanitize_for_numba(symptoms_to_hospital_days), _sanitize_for_numba(hospitalization_length_of_stay_general), _sanitize_for_numba(hospitalization_length_of_stay_icu), _sanitize_for_numba(hospitalization_length_of_stay_icu_and_ventilator), _sanitize_for_numba(fraction_icu_requiring_ventilator), _sanitize_for_numba(mortality_rate_from_ICU), _sanitize_for_numba(mortality_rate_from_hospital), _sanitize_for_numba(mortality_rate_no_ICU_beds), _sanitize_for_numba(mortality_rate_from_ICUVent), _sanitize_for_numba(mortality_rate_no_general_beds), _sanitize_for_numba(beds_general), _sanitize_for_numba(beds_ICU)))

@numba.njit
def rk4_step(y, t, dt, suppression_value, params):
    k1 = seir_time_step(y, t, suppression_value, params)
    k2 = seir_time_step(y + 0.5 * dt * k1, t + 0.5 * dt, suppression_value, params)
    k3 = seir_time_step(y + 0.5 * dt * k2, t + 0.5 * dt, suppression_value, params)
    k4 = seir_time_step(y + dt * k3, t + dt, suppression_value, params)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

@numba.njit
def run_seir(t_list, y0, suppression_value, params):
    n = len(t_list)
    result_time_series = np.zeros((n, 12))
    result_time_series[0] = y0
    for i in range(1, n):
        dt = t_list[i] - t_list[i - 1]
        result_time_series[i] = rk4_step(result_time_series[i - 1], t_list[i - 1], dt, suppression_value, params)
    S = np.zeros(n)
    E = np.zeros(n)
    A = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    HGen = np.zeros(n)
    HICU = np.zeros(n)
    HICUVent = np.zeros(n)
    D = np.zeros(n)
    HAdmissions_general = np.zeros(n)
    HAdmissions_ICU = np.zeros(n)
    TotalAllInfections = np.zeros(n)
    for i in range(n):
        S[i] = result_time_series[i, 0]
        E[i] = result_time_series[i, 1]
        A[i] = result_time_series[i, 2]
        I[i] = result_time_series[i, 3]
        R[i] = result_time_series[i, 4]
        HGen[i] = result_time_series[i, 5]
        HICU[i] = result_time_series[i, 6]
        HICUVent[i] = result_time_series[i, 7]
        D[i] = result_time_series[i, 8]
        HAdmissions_general[i] = result_time_series[i, 9]
        HAdmissions_ICU[i] = result_time_series[i, 10]
        TotalAllInfections[i] = result_time_series[i, 11]
    direct_deaths_per_day = derivative(D)
    beds_general = params[IDX_BEDS_GEN]
    beds_icu = params[IDX_BEDS_ICU]
    mort_no_gen = params[IDX_MORT_NO_GEN]
    mort_no_icu = params[IDX_MORT_NO_ICU]
    los_gen = params[IDX_LOS_GEN]
    los_icu = params[IDX_LOS_ICU]
    los_icu_vent = params[IDX_LOS_ICU_VENT]
    deaths_from_hospital_bed_limits = np.zeros(n)
    deaths_from_icu_bed_limits = np.zeros(n)
    HGen_cumulative = np.zeros(n)
    HICU_cumulative = np.zeros(n)
    HVent_cumulative = np.zeros(n)
    sum_hgen_death = 0.0
    sum_hicu_death = 0.0
    sum_hgen = 0.0
    sum_hicu = 0.0
    sum_hvent = 0.0
    for i in range(n):
        val_gen = HGen[i] - beds_general
        if val_gen > 0:
            sum_hgen_death += val_gen * mort_no_gen / los_gen
        deaths_from_hospital_bed_limits[i] = sum_hgen_death
        val_icu = HICU[i] - beds_icu
        if val_icu > 0:
            sum_hicu_death += val_icu * mort_no_icu / los_icu
        deaths_from_icu_bed_limits[i] = sum_hicu_death
        sum_hgen += HGen[i]
        HGen_cumulative[i] = sum_hgen / los_gen
        sum_hicu += HICU[i]
        HICU_cumulative[i] = sum_hicu / los_icu
        sum_hvent += HICUVent[i]
        HVent_cumulative[i] = sum_hvent / los_icu_vent
    total_new_infections = derivative(TotalAllInfections)
    total_deaths_per_day = derivative(D)
    results = np.zeros((17, n))
    for i in range(n):
        results[0, i] = S[i]
        results[1, i] = E[i]
        results[2, i] = A[i]
        results[3, i] = I[i]
        results[4, i] = R[i]
        results[5, i] = HGen[i]
        results[6, i] = HICU[i]
        results[7, i] = HICUVent[i]
        results[8, i] = D[i]
        results[9, i] = direct_deaths_per_day[i]
        results[10, i] = deaths_from_hospital_bed_limits[i]
        results[11, i] = deaths_from_icu_bed_limits[i]
        results[12, i] = HGen_cumulative[i]
        results[13, i] = HICU_cumulative[i]
        results[14, i] = HVent_cumulative[i]
        results[15, i] = total_new_infections[i]
        results[16, i] = total_deaths_per_day[i]
    return results

@numba.njit
def _steady_state_ratios_jit(r, suppression, N, params):
    tlist = np.linspace(0.0, 60.0, 61)
    if r > 1:
        initial_infected = 100.0
    else:
        initial_infected = 10000.0
    A_initial = 0.7 * initial_infected
    I_initial = initial_infected
    S_initial = N - A_initial - I_initial
    y0 = np.zeros(12)
    y0[0] = S_initial
    y0[2] = A_initial
    y0[3] = I_initial
    results = run_seir(tlist, y0, suppression, params)
    E_final = results[1, -1]
    I_final = results[3, -1]
    A_final = results[2, -1]
    HGen_final = results[5, -1]
    HICU_final = results[6, -1]
    HICUVent_final = results[7, -1]
    direct_deaths_final = results[9, -1]
    ratios = np.zeros(7)
    ratios[0] = E_final / I_final
    ratios[1] = 1.0
    ratios[2] = A_final / I_final
    ratios[3] = HGen_final / I_final
    ratios[4] = HICU_final / I_final
    ratios[5] = HICUVent_final / I_final
    ratios[6] = direct_deaths_final / I_final
    return ratios

def steady_state_ratios(r, suppression, N, params):
    return _deep_to_list(_steady_state_ratios_jit(_sanitize_for_numba(r), _sanitize_for_numba(suppression), _sanitize_for_numba(N), _sanitize_for_numba(params)))