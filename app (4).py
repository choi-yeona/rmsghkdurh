"""app.py  -  수렴반경 탐험기 (Streamlit)

탐구 질문: 매끈한 함수 1/(1+x^2) 의 테일러 급수는 왜 어떤 구간에서만 수렴할까?
가설    : 수렴반경 R 은 전개 중심 a 에서 복소평면의 특이점 x = ±i 까지의 거리, 즉 R = sqrt(a^2+1).

계산은 series_core.py, 화면 구성은 이 파일이 맡습니다.
실행:  streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import series_core as sc

st.set_page_config(page_title="수렴반경 탐험기", page_icon="📐", layout="wide")

N_MAX = 150  # 계수를 미리 계산해 두는 최대 항 수


# ----------------------------------------------------------------------
# 계산 결과 캐시 (같은 입력이면 다시 계산하지 않음)
# ----------------------------------------------------------------------
@st.cache_data
def get_coeffs(func, a, n_max):
    return sc.taylor_coeffs(func, a, n_max)


@st.cache_data
def radius_table(n_est, a_values):
    """여러 중심 a 에서 (추정 R, 이론 R) 계산."""
    rows = []
    for a in a_values:
        c = sc.coef_recurrence(a, n_est)
        rows.append((a, sc.radius_estimate(c), sc.singularity_radius(a)))
    return rows


# ----------------------------------------------------------------------
# 그림 그리는 함수들 (그림 글자는 서버에 한글 폰트가 없어도 되도록 영어)
# ----------------------------------------------------------------------
def plot_real_line(func, c, a, N, x0, R):
    lo = min(a - R, x0, a) - 1.5
    hi = max(a + R, x0, a) + 1.5
    x = np.linspace(lo, hi, 900)
    truth = sc.f_true(func, x)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(a - R, a + R, color="green", alpha=0.12, label="convergence interval (a-R, a+R)")
    ax.plot(x, truth, "k", lw=3, label="true function")
    for n_show, lw, alpha in [(N // 4, 1.2, 0.5), (N // 2, 1.2, 0.7), (N, 2.2, 1.0)]:
        if n_show >= 1:
            ax.plot(x, sc.partial_sum(c, x, a, n_show), lw=lw, alpha=alpha,
                    label=f"partial sum N={n_show}")
    ax.axvline(a, color="gray", ls=":", lw=1, label=f"center a={a:.2f}")
    ax.axvline(x0, color="red", ls="--", lw=1.3, label=f"x0={x0:.2f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(truth.min() - 1.5, truth.max() + 1.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    fig.tight_layout()
    return fig


def plot_complex_plane(a, R, x0, inside):
    th = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(a + R * np.cos(th), R * np.sin(th), color="tab:green", lw=2,
            label=f"circle of radius R={R:.3f}")
    ax.plot([a], [0], "o", color="tab:blue", ms=9, label=f"center a={a:.2f}")
    ax.plot([0, 0], [1, -1], "x", color="red", ms=12, mew=3, label="singularities x=+i, -i")
    ax.plot([a, 0], [0, 1], "r--", lw=1)
    ax.plot([a, 0], [0, -1], "r--", lw=1)
    ax.plot([x0], [0], "*", ms=16, color="tab:green" if inside else "tab:red",
            label=f"x0={x0:.2f} ({'inside' if inside else 'outside'})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(min(a - R, x0, -1) - 0.7, max(a + R, x0, 1) + 0.7)
    ax.set_ylim(-(R + 0.7), R + 0.7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_convergence_at_point(S, true_value, rho_theory, n_max):
    n = np.arange(len(S))
    err = np.maximum(np.abs(S - true_value), 1e-17)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    # 왼쪽: 부분합 자체
    ax1.plot(n, S, "o-", ms=3, color="tab:blue", label="partial sum S_n(x0)")
    ax1.axhline(true_value, color="k", lw=2, label="true value f(x0)")
    if np.all(np.isfinite(S)):
        lim = max(abs(true_value) * 3, 3)
        ax1.set_ylim(true_value - lim, true_value + lim)
    ax1.set_xlabel("n (number of terms)")
    ax1.set_ylabel("S_n(x0)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("partial sums")

    # 오른쪽: 오차 (로그 눈금) + 이론 기울기
    ax2.semilogy(n, err, "o-", ms=3, color="tab:red", label="|S_n - f(x0)|")
    n0 = 5
    if len(err) > n0 + 8 and rho_theory > 0:
        e0 = err[n0:n0 + 8].max()
        ref = e0 * rho_theory ** (n[n0:] - n0)
        ax2.semilogy(n[n0:], np.maximum(ref, 1e-17), "k--", lw=1.5,
                     label=f"theory: rho^n,  rho=|x0-a|/R={rho_theory:.3f}")
    ax2.set_xlabel("n (number of terms)")
    ax2.set_ylabel("error (log scale)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    ax2.set_title("error vs n")
    fig.tight_layout()
    return fig


def plot_radius_vs_center(rows):
    a_vals = np.array([r[0] for r in rows])
    est = np.array([r[1] for r in rows])
    aa = np.linspace(0, a_vals.max(), 300)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(aa, np.sqrt(aa ** 2 + 1), "k-", lw=2, label="theory  R = sqrt(a^2+1)")
    ax.plot(a_vals, est, "o", color="tab:red", label="estimated from coefficients")
    ax.plot(aa, aa, "--", color="gray", lw=1, label="R = a  (for comparison)")
    ax.set_xlabel("center a")
    ax.set_ylabel("radius of convergence R")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_vs_N(a, Ns, rel_errors):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    nn = np.linspace(min(Ns), max(Ns), 200)
    ax.plot(nn, [sc.predicted_relative_error(a, n) * 100 for n in nn], "k--",
            label="prediction  ln(sqrt(1+a^2)) / N")
    ax.plot(Ns, np.array(rel_errors) * 100, "o-", color="tab:red", label="measured")
    ax.set_xlabel("number of terms N used for estimate")
    ax.set_ylabel("relative error of R (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 화면
# ----------------------------------------------------------------------
st.title("📐 수렴반경 탐험기")
st.caption("매끈한 함수의 테일러 급수가 왜 어떤 구간에서만 수렴할까?  가설: 수렴반경 R = 중심에서 복소평면 특이점 ±i 까지의 거리 = √(a²+1)")

with st.sidebar:
    st.header("설정")
    func_label = st.selectbox("함수 f(x)", list(sc.FUNCTIONS))
    func = sc.FUNCTIONS[func_label]
    a = st.slider("전개 중심 a", 0.0, 4.0, 0.0, 0.1)
    N = st.slider("부분합 항 수 N", 1, 60, 10)
    x0 = st.number_input("관찰할 점 x₀", min_value=-8.0, max_value=8.0, value=2.0, step=0.1)
    st.markdown("---")
    st.caption("두 함수 모두 특이점이 x = ±i 로 같아서 수렴반경도 같아야 합니다.")

R = sc.singularity_radius(a)
dist = abs(x0 - a)
inside = dist < R - 1e-9
c = get_coeffs(func, a, N_MAX)
true_x0 = float(sc.f_true(func, x0))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 실수 그래프", "② 복소평면", "③ x₀에서의 수렴", "④ 수렴반경 검증", "⑤ 해설"]
)

# ---------------- ① 실수 그래프 ----------------
with tab1:
    S_N = float(sc.partial_sum(c, x0, a, N))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("수렴반경 R = √(a²+1)", f"{R:.4f}")
    m2.metric("거리 |x₀ − a|", f"{dist:.4f}")
    if abs(dist - R) < 1e-9:
        verdict = "경계 (판단 어려움)"
    elif inside:
        verdict = "구간 안 → 수렴 예상"
    else:
        verdict = "구간 밖 → 발산 예상"
    m3.metric("판정", verdict)
    m4.metric("진짜 값 f(x₀)", f"{true_x0:.6f}")
    m5.metric(f"부분합 S_{N}(x₀)", f"{S_N:.6g}", delta=f"오차 {abs(S_N - true_x0):.3g}",
              delta_color="off")

    fig = plot_real_line(func, c, a, N, x0, R)
    st.pyplot(fig)
    plt.close(fig)
    st.info("초록 음영이 이론상 수렴 구간입니다. 항 수 N을 늘리면 음영 안에서는 부분합이 진짜 함수에 달라붙고, "
            "음영 밖에서는 오히려 더 크게 어긋납니다. 사이드바에서 중심 a를 움직여 보세요.")

# ---------------- ② 복소평면 ----------------
with tab2:
    left, right = st.columns([1, 1])
    with left:
        fig = plot_complex_plane(a, R, x0, inside)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("### 이 그림이 말해주는 것")
        st.markdown(
            f"""
- 빨간 ×는 분모 $1+x^2$ 가 0이 되는 **복소수 점 $x=\\pm i$** 입니다. 실수 그래프에서는 보이지 않습니다.
- 초록 원은 **중심 $a={a:.2f}$ 에서 가장 가까운 문제점까지** 거리 $R=\\sqrt{{a^2+1}}={R:.3f}$ 를 반지름으로 한 원입니다.
- 별(★)은 관찰하는 점 $x_0={x0:.2f}$ 입니다. 원 **안**이면 초록, **밖**이면 빨강입니다.
- 현재 $|x_0-a|={dist:.3f}$, $R={R:.3f}$ → **{'원 안: 급수가 수렴' if inside else '원 밖: 급수가 발산'}**
"""
        )
        st.caption("중심 a가 0에서 커질수록 문제점 ±i 에서 멀어져 원이 커집니다. 그래서 같은 점 x₀ 가 원 안으로 들어올 수 있습니다.")

# ---------------- ③ x0 에서의 수렴 ----------------
with tab3:
    n_plot = st.slider("그릴 최대 항 수", 10, N_MAX, 60, key="n_plot")
    S = sc.partial_sums_at(c[: n_plot + 1], x0, a)
    rho_theory = dist / R
    rho_est = sc.error_decay_ratio(S, true_x0)

    k1, k2, k3 = st.columns(3)
    k1.metric("이론 감소율 ρ = |x₀−a| / R", f"{rho_theory:.4f}")
    k2.metric("측정한 감소율(항마다 곱해지는 배수)", "계산 불가" if rho_est is None else f"{rho_est:.4f}")
    k3.metric("결론", "수렴 (ρ<1)" if rho_theory < 1 else ("발산 (ρ>1)" if rho_theory > 1 else "경계 (ρ=1)"))

    fig = plot_convergence_at_point(S, true_x0, rho_theory, n_plot)
    st.pyplot(fig)
    plt.close(fig)
    st.info("오차가 항마다 약 ρ배씩 변합니다. ρ<1이면 오차가 로그 눈금에서 직선으로 내려가고(수렴), "
            "ρ>1이면 직선으로 올라갑니다(발산). 검은 점선이 이론 기울기입니다.")
    if rho_est is None:
        st.caption("x₀ = a 이면 첫 항에서 이미 정확해 오차가 0이거나, 오차가 너무 빨리 반올림 한계(약 1e-11)에 닿아 측정이 불가능할 수 있습니다.")
    if func == "arctan":
        st.caption("arctan은 계수에 1/n이 붙어서 측정한 감소율이 이론값보다 조금 작게 나옵니다(n이 클수록 이론값에 가까워짐).")

# ---------------- ④ 수렴반경 검증 ----------------
with tab4:
    st.subheader("(1) 두 가지 계수 계산법이 일치하는가?")
    c_rec = sc.coef_recurrence(a, 60)
    c_cls = sc.coef_closed(a, 60)
    rel = float(np.max(np.abs(c_rec - c_cls) / (np.abs(c_cls) + 1e-300)))
    v1, v2 = st.columns(2)
    v1.metric("점화식 vs 복소수 닫힌 식 (최대 상대차이)", f"{rel:.2e}")
    v2.metric("판정", "일치 (계산 오차 수준)" if rel < 1e-9 else "불일치 — 확인 필요")
    st.caption("1/(1+x²) 의 계수를 서로 다른 두 방법으로 계산해 비교합니다 (현재 중심 a 기준, 60개 항).")

    st.subheader("(2) 계수에서 추정한 수렴반경 vs 이론 √(a²+1)")
    n_est = st.slider("추정에 쓸 항 수 N", 50, 300, 150, 10, key="n_est")
    a_grid = tuple(float(v) for v in np.linspace(0, 5, 21))  # 0, 0.25, 0.5, ..., 5
    rows = radius_table(n_est, a_grid)
    g1, g2 = st.columns([1, 1])
    with g1:
        fig = plot_radius_vs_center(rows)
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        pick = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        table = {"중심 a": [], "추정 R": [], "이론 √(a²+1)": [], "상대오차(%)": [], "예측 오차(%)": []}
        for a_i, r_est, r_th in rows:
            if a_i in pick:
                table["중심 a"].append(a_i)
                table["추정 R"].append(round(r_est, 4))
                table["이론 √(a²+1)"].append(round(r_th, 4))
                table["상대오차(%)"].append(round(abs(r_est - r_th) / r_th * 100, 3))
                table["예측 오차(%)"].append(round(sc.predicted_relative_error(a_i, n_est) * 100, 3))
        st.table(table)
    st.caption("추정값은 항상 이론값보다 약간 큽니다. 오차는 대략 ln(√(1+a²))/N 으로 예측되며(마지막 열), 항 수 N이 커지면 줄어듭니다.")

    st.subheader("(3) 항 수 N을 늘리면 추정이 좋아질까? (현재 중심 a 기준)")
    Ns = [20, 40, 80, 160, 320]
    theory_R = sc.singularity_radius(a)
    est_list = [sc.radius_estimate(sc.coef_recurrence(a, n)) for n in Ns]
    rel_errors = [(e - theory_R) / theory_R for e in est_list]
    h1, h2 = st.columns([1, 1])
    with h1:
        st.table({"N": Ns,
                  "추정 R": [round(e, 4) for e in est_list],
                  "상대오차(%)": [round(e * 100, 3) for e in rel_errors]})
        st.caption(f"이론값 R = {theory_R:.4f}.  N이 2배가 되면 오차가 약 절반이 됩니다(오차 ∝ 1/N).")
    with h2:
        fig = plot_error_vs_N(a, Ns, rel_errors)
        st.pyplot(fig)
        plt.close(fig)

# ---------------- ⑤ 해설 ----------------
with tab5:
    st.markdown("## 이 앱은 무엇을 계산하나요?")
    st.markdown("함수 $f(x)=\\dfrac{1}{1+x^2}$ 를 중심 $a$ 에서 $f(x)=\\sum_{n\\ge 0} c_n (x-a)^n$ 으로 전개했을 때, "
                "급수가 어디까지 맞는지(수렴반경)를 계수 $c_n$ 으로부터 알아냅니다.")

    st.markdown("### 1. 계수 계산 (방법 ①: 점화식)")
    st.markdown("$(1+x^2)f=1$ 에서 $x=a+t$ 로 두고 $t^n$ 의 계수를 비교합니다.")
    st.latex(r"(1+a^2)c_0=1,\quad (1+a^2)c_1+2a\,c_0=0,\quad (1+a^2)c_n+2a\,c_{n-1}+c_{n-2}=0\;(n\ge2)")

    st.markdown("### 2. 수렴반경의 증명")
    st.markdown("$c_n\\approx r^n$ 이라 놓으면 특성방정식이 나옵니다.")
    st.latex(r"(1+a^2)r^2+2a\,r+1=0\;\Rightarrow\; r=\frac{-a\pm i}{1+a^2},\qquad |r|=\frac{1}{\sqrt{1+a^2}}")
    st.markdown("판별식이 $-4<0$ 이므로 **복소수 근**이 나오고, $|c_n|\\approx |r|^n$ 이므로 급수는 "
                "$|x-a|<\\sqrt{1+a^2}$ 에서 수렴합니다. $\\sqrt{1+a^2}=|a-i|$ 는 중심에서 $i$ 까지의 거리입니다.")

    st.markdown("### 3. 검증용 닫힌 식 (방법 ②)")
    st.latex(r"c_n=(-1)^n\,\mathrm{Im}\!\left[\frac{1}{(a-i)^{n+1}}\right]")
    st.markdown("부분분수 $\\frac{1}{1+x^2}=\\frac{1}{2i}\\left(\\frac{1}{x-i}-\\frac{1}{x+i}\\right)$ 와 등비급수로 얻습니다. "
                "점화식과 전혀 다른 계산인데 같은 값을 주는지 탭 ④(1)에서 확인합니다.")

    st.markdown("### 4. 탭별로 무엇을 보면 되나요?")
    st.markdown(
        """
| 탭 | 볼 것 | 얻는 결론 |
|---|---|---|
| ① 실수 그래프 | 부분합 곡선과 초록 수렴 구간 | 구간 안에서는 맞고 밖에서는 튄다 |
| ② 복소평면 | 중심에서 ±i 에 닿는 원 | 원의 반지름이 수렴반경 |
| ③ x₀에서의 수렴 | 오차의 로그 그래프 기울기 | 항마다 오차가 \\|x₀−a\\|/R 배 |
| ④ 수렴반경 검증 | 추정 R vs 이론 R | 가설이 수치로 확인됨 |
"""
    )

    st.markdown("### 5. 한계와 주의")
    st.markdown(
        """
- 계수로 추정한 수렴반경은 이론값보다 **항상 약간 큽니다.** 오차는 $\\ln\\sqrt{1+a^2}/N$ 정도이며 항 수가 늘면 줄어듭니다.
- 계수가 복소수 근 때문에 진동하므로 마지막 항 하나가 아니라 **최근 12개 중 최댓값**을 사용합니다.
- 오차 감소율 측정은 오차가 컴퓨터 반올림 한계(약 1e-11)에 닿기 전 구간만 사용합니다.
- 이 앱은 중심 $a$ 가 실수축 위에 있는 경우만 다룹니다.
"""
    )
    st.markdown("### 6. 다음 단계")
    st.markdown("수렴 구간 밖에서는 정말 끝일까? **파데 근사**(분수 꼴 근사)로 발산 구간의 값을 복원할 수 있는지 탐구해 볼 수 있습니다.")
