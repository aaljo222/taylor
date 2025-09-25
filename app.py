import time
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Taylor Series Animator", page_icon="📈", layout="wide")

st.title("📈 Taylor Series Animation (Streamlit)")
st.markdown(
    """
Animate how a Taylor polynomial approaches a target function.
Use the controls in the sidebar and click **Start Animation**.
"""
)

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    func_name = st.selectbox("Target function f(x)", ["sin(x)", "cos(x)", "exp(x)"])
    center = st.number_input("Expansion center a", value=0.0, step=0.5)
    max_order = st.slider("Max polynomial order (N)", 1, 30, 12)
    x_min, x_max = st.slider("x-range", -10.0, 10.0, (-6.0, 6.0))
    num_points = st.select_slider("Resolution (points)", options=[200, 400, 800, 1200], value=800)
    fps = st.select_slider("Animation speed (frames/sec)", options=[2, 4, 8, 12, 24], value=8)
    pause_end = st.checkbox("Pause 1s at the end", value=True)
    animate = st.button("▶️ Start Animation")
    show_error = st.checkbox("Show absolute error curve |f - T_N|", value=True)

# --- Choose function and its derivatives ---
def target_and_derivs(name):
    if name == "sin(x)":
        f = np.sin
        # derivatives cycle: sin, cos, -sin, -cos, ...
        def nth_derivative_at_a(n, a):
            k = n % 4
            if k == 0:
                return np.sin(a)
            if k == 1:
                return np.cos(a)
            if k == 2:
                return -np.sin(a)
            return -np.cos(a)
        return f, nth_derivative_at_a

    if name == "cos(x)":
        f = np.cos
        # derivatives cycle: cos, -sin, -cos, sin, ...
        def nth_derivative_at_a(n, a):
            k = n % 4
            if k == 0:
                return np.cos(a)
            if k == 1:
                return -np.sin(a)
            if k == 2:
                return -np.cos(a)
            return np.sin(a)
        return f, nth_derivative_at_a

    # exp(x)
    f = np.exp
    def nth_derivative_at_a(n, a):
        return np.exp(a)  # all derivatives are e^a
    return f, nth_derivative_at_a

f, fderiv = target_and_derivs(func_name)

# --- Precompute x-grid and true function ---
x = np.linspace(x_min, x_max, int(num_points))
fx = f(x)

# --- Build Taylor coefficients up to max_order at 'center' ---
coeffs = np.array([fderiv(n, center) / math.factorial(n) for n in range(max_order + 1)], dtype=float)

def taylor_poly_vals(x, a, coeffs, N):
    # Evaluate T_N(x) = sum_{n=0..N} c_n (x-a)^n
    dx = x - a
    # Horner-like evaluation for speed
    y = np.zeros_like(x)
    for n in range(N, -1, -1):
        y = y * dx + coeffs[n]
    return y

# --- Plot containers ---
col1, col2 = st.columns([2, 1])
plot_ph = col1.empty()
err_ph = col2.empty()

# --- Static initial plot (N=0) ---
N0 = 0
y0 = taylor_poly_vals(x, center, coeffs, N0)

def draw(N):
    yN = taylor_poly_vals(x, center, coeffs, N)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, fx, label="f(x)")
    ax1.plot(x, yN, label=f"T_{N}(x) around a={center}")
    ax1.axvline(center, linestyle="--", linewidth=1)
    ax1.set_title(f"{func_name} vs Taylor Polynomial (order N={N})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plot_ph.pyplot(fig1)
    plt.close(fig1)

    if show_error:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.plot(x, np.abs(fx - yN), label="|f(x) - T_N(x)|")
        ax2.set_title("Absolute Error")
        ax2.set_xlabel("x")
        ax2.set_ylabel("error")
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        err_ph.pyplot(fig2)
        plt.close(fig2)
    else:
        err_ph.empty()

# Draw initial
draw(N0)

# --- Animate if requested ---
if animate:
    delay = 1.0 / max(fps, 1)
    for N in range(1, max_order + 1):
        draw(N)
        time.sleep(delay)
    if pause_end:
        time.sleep(1.0)

# --- Theory box ---
with st.expander("🧠 Quick theory: Taylor series"):
    st.markdown(r"""
A Taylor polynomial of order \(N\) for \(f(x)\) around \(a\) is
\[
T_N(x) = \sum_{n=0}^{N} rac{f^{(n)}(a)}{n!}\,(x-a)^n.
\]
For analytic functions (e.g., \(e^x, \sin x, \cos x\)), \(T_N(x)\) converges to \(f(x)\) as \(N 	o \infty\) for all \(x\).
""")
