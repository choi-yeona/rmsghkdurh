"""series_core.py

수렴반경 탐험기의 "수식 계산" 부분만 모아 둔 모듈입니다.
화면(Streamlit)과 분리해 두면 계산이 맞는지 따로 테스트하기 쉽습니다.

다루는 함수
  - rational : f(x) = 1/(1+x^2)
  - arctan   : f(x) = arctan(x)   (도함수가 1/(1+x^2) 이라서 같은 특이점 ±i 를 가짐)
"""
import numpy as np

# 화면에 보여 줄 이름 -> 내부 이름
FUNCTIONS = {
    "1/(1+x²)": "rational",
    "arctan x": "arctan",
}


# ----------------------------------------------------------------------
# 1. 계수 계산
# ----------------------------------------------------------------------
def coef_recurrence(a, N):
    """방법 ①: 점화식으로 1/(1+x^2) 의 중심 a 테일러 계수 c_0..c_N 계산.

    (1+x^2)*f = 1 에서 x = a + t 로 두고 t^n 의 계수를 비교하면
        (1+a^2) c_0 = 1
        (1+a^2) c_1 + 2a c_0 = 0
        (1+a^2) c_n + 2a c_{n-1} + c_{n-2} = 0   (n >= 2)
    """
    c = np.zeros(N + 1)
    c[0] = 1 / (1 + a * a)
    if N >= 1:
        c[1] = -2 * a * c[0] / (1 + a * a)
    for n in range(2, N + 1):
        c[n] = -(2 * a * c[n - 1] + c[n - 2]) / (1 + a * a)
    return c


def coef_closed(a, N):
    """방법 ②: 복소수 닫힌 식 c_n = (-1)^n * Im[ 1/(a - i)^(n+1) ] (검증용)."""
    return np.array([((-1) ** n) * ((a - 1j) ** (-(n + 1))).imag
                     for n in range(N + 1)])


def taylor_coeffs(func, a, N):
    """선택한 함수의 중심 a 테일러 계수 c_0..c_N.

    arctan 은 1/(1+x^2) 을 적분한 함수라서
        arctan(a+t) = arctan(a) + sum_{n>=1} (g_{n-1} / n) t^n
    (g 는 1/(1+x^2) 의 계수) 로 얻습니다.
    """
    if func == "rational":
        return coef_recurrence(a, N)
    if func == "arctan":
        g = coef_recurrence(a, max(N - 1, 0))
        c = np.zeros(N + 1)
        c[0] = np.arctan(a)
        for n in range(1, N + 1):
            c[n] = g[n - 1] / n
        return c
    raise ValueError(f"unknown function: {func}")


# ----------------------------------------------------------------------
# 2. 함수값, 부분합
# ----------------------------------------------------------------------
def f_true(func, x):
    x = np.asarray(x, dtype=float)
    if func == "rational":
        return 1 / (1 + x ** 2)
    if func == "arctan":
        return np.arctan(x)
    raise ValueError(f"unknown function: {func}")


def partial_sum(c, x, a, N):
    """부분합 S_N(x) = sum_{n=0}^{N} c_n (x-a)^n  (x 는 배열도 가능)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        return sum(c[n] * (x - a) ** n for n in range(N + 1))


def partial_sums_at(c, x0, a):
    """한 점 x0 에서 S_0, S_1, ..., S_N 을 한꺼번에 (누적합)."""
    t = x0 - a
    with np.errstate(all="ignore"):
        terms = c * t ** np.arange(len(c))
    return np.cumsum(terms)


# ----------------------------------------------------------------------
# 3. 수렴반경
# ----------------------------------------------------------------------
def singularity_radius(a):
    """이론값: 중심 a 에서 특이점 +-i 까지의 거리 = sqrt(a^2 + 1)."""
    return float(np.sqrt(a * a + 1))


def radius_estimate(c, window=12):
    """계수로 수렴반경 추정 (Cauchy-Hadamard 의 유한 버전).

    마지막 window 개 계수의 |c_k|^(1/k) 중 최댓값(limsup 대신)으로 R = 1/최댓값.
    """
    N = len(c) - 1
    vals = [abs(c[k]) ** (1 / k)
            for k in range(max(N - window + 1, 1), N + 1) if abs(c[k]) > 0]
    return 1 / max(vals)


def predicted_relative_error(a, N):
    """추정 오차의 이론적 크기 ln(sqrt(1+a^2)) / N (분석으로 얻은 근사식)."""
    return float(np.log(np.sqrt(1 + a * a)) / N)


def error_decay_ratio(partial_sums, true_value, window=8, skip=5):
    """부분합 오차 |S_n - f| 가 항마다 몇 배씩 변하는지(ratio) 추정.

    오차가 진동하며 0 근처를 지나므로, window 칸짜리 최댓값(포락선)에
    로그를 취해 직선으로 맞춥니다.  ratio < 1 이면 수렴, > 1 이면 발산.
    계산할 수 없으면 None.
    """
    err = np.abs(np.asarray(partial_sums) - true_value)
    if len(err) < skip + 2 * window or not np.all(np.isfinite(err)):
        return None
    env = np.array([err[k:k + window].max() for k in range(skip, len(err) - window)])

    # 오차가 컴퓨터 반올림 오차(약 1e-16) 근처까지 내려가면 그 뒤는 의미가 없으므로
    # 바닥(floor)보다 큰 구간만 사용합니다.
    floor = 1e-11 * max(1.0, abs(float(true_value)))
    below = np.where(env < floor)[0]
    end = int(below[0]) if len(below) else len(env)
    if end < 8:
        return None
    ks = np.arange(end)
    slope = np.polyfit(ks, np.log(env[:end]), 1)[0]
    return float(np.exp(slope))
