"""
Example dash browser application visualizing basic diffusion model training/sampling.
Uses a basic Random Fourier Features model to keep things simple.

TODO: use a more appropriate CSS fading effect -- local assets folder [less priority]
TODO: a better JAX-powered version ? Different project "djaxsh" <- using PFGM++ machinery maybe?
      (save this for another project focusing on trajectory diffusion -- more important anyway)

"""

from dash import Dash, dcc, html, Input, Output, State, callback, ctx, no_update
import uuid
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from flask_caching import Cache
from lsrff import LSRFF

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",  # Dash CSS
    "https://codepen.io/chriddyp/pen/brPBPO.css",  # Loading screen CSS
]

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)

app.title = "Diffusion Illustrator"

cache = Cache(
    app.server,
    config={
        # "CACHE_TYPE": "FileSystemCache",
        # "CACHE_DIR": "cache-directory",
        "CACHE_TYPE": "SimpleCache",
        "CACHE_THRESHOLD": 20,
        "CACHE_DEFAULT_TIMEOUT": int(1000000),
    },
)

STATE_CACHE_VARIABLE_NAME = "state"


def get_session_cache(sesh: str, name: str):
    return cache.get(sesh + ":" + name)


def set_session_cache(sesh: str, name: str, obj):
    cache.set(sesh + ":" + name, obj)


def create_default_sesh_state() -> dict:
    return {
        "schedule": "loglinear",
        "T": int(150),
        "log2-beta-min": -8.0,
        "log2-beta-max": -2.0,
        "B": int(100),  # forward histogram bins
        "BB": int(100),  # backward histogram bins
        "xmin": -4.0,
        "xmax": 4.0,
        "FRWD": np.tile(np.nan, (1, 1)),
        "Model": LSRFF(400, 2, 1, np.ones(2)),
        "BKWD": np.tile(np.nan, (1, 1)),
        "bkwd-noise-level": 1.0,
    }


def get_schedule(STATE) -> np.array:
    if STATE["schedule"] == "linear":
        return np.linspace(
            np.power(2.0, STATE["log2-beta-min"]),
            np.power(2.0, STATE["log2-beta-max"]),
            STATE["T"],
        )
    elif STATE["schedule"] == "loglinear":
        return np.power(
            2.0, np.linspace(STATE["log2-beta-min"], STATE["log2-beta-max"], STATE["T"])
        )
    elif STATE["schedule"] == "sigma":
        return (
            np.linspace(
                np.sqrt(np.power(2.0, STATE["log2-beta-min"])),
                np.sqrt(np.power(2.0, STATE["log2-beta-max"])),
                STATE["T"],
            )
            ** 2
        )
    elif STATE["schedule"] == "phase":
        phi0 = np.arccos(np.sqrt(1 - np.power(2.0, STATE["log2-beta-min"])))
        phi1 = np.arccos(np.sqrt(1 - np.power(2.0, STATE["log2-beta-max"])))
        return np.sin(np.linspace(phi0, phi1, STATE["T"])) ** 2
    elif STATE["schedule"] == "logphase":
        phi0 = np.arccos(np.sqrt(1 - np.power(2.0, STATE["log2-beta-min"])))
        phi1 = np.arccos(np.sqrt(1 - np.power(2.0, STATE["log2-beta-max"])))
        phase = np.exp(np.linspace(np.log(phi0), np.log(phi1), STATE["T"]))
        return np.sin(phase) ** 2
    else:
        raise NotImplementedError


def get_coefs(STATE) -> dict:
    beta_seq = get_schedule(STATE)
    assert np.all(beta_seq >= 0) and np.all(beta_seq <= 1)
    lmbda_seq = np.sqrt(1 - beta_seq)
    sigma_seq = np.sqrt(beta_seq)
    C_seq = np.cumprod(lmbda_seq)
    D_seq = np.sqrt(1.0 - C_seq * C_seq)
    A_seq = (C_seq / lmbda_seq) * (1 - lmbda_seq**2) / (D_seq**2)
    B_seq = (1.0 - C_seq**2 / (lmbda_seq**2)) * lmbda_seq / (D_seq**2)
    SNR_seq = (C_seq**2) / (D_seq**2)
    return {
        "lambda": lmbda_seq,
        "sigma": sigma_seq,
        "C": C_seq,
        "D": D_seq,
        "A": A_seq,
        "B": B_seq,
        "SNR": SNR_seq,
    }


def get_bin_edges(STATE, name: str = "B") -> np.array:
    return np.linspace(STATE["xmin"], STATE["xmax"], STATE[name] + 1)


def get_bin_centers(STATE, name: str = "B") -> np.array:
    edges = get_bin_edges(STATE, name=name)
    return np.array([(edges[k] + edges[k + 1]) / 2 for k in range(len(edges) - 1)])


def add_forward_paths(
    C: np.array, edges: np.array, beta_seq: np.array, X0: np.array
) -> None:
    T = len(beta_seq)
    E = len(edges)
    assert C.shape[1] == T
    assert C.shape[0] == E - 1
    M = len(X0)  # number of samples
    X = np.copy(X0)
    for t in range(T):
        lmbda = np.sqrt(1 - beta_seq[t])
        sigma = np.sqrt(beta_seq[t])
        X = lmbda * X + sigma * np.random.randn(M)
        counts_, edges_ = np.histogram(X, bins=edges)
        assert np.all(edges_ == edges)
        C[:, t] = C[:, t] + counts_


def get_source_sample(M: int) -> np.array:
    return np.concatenate(
        [
            2 + np.random.rand(M // 3),
            -3 + np.random.rand(M // 3),
            -0.5 + np.random.rand(M - 2 * (M // 3)),
        ]
    )


def sample_ddpm_batch(STATE: dict, B: int):
    x0 = get_source_sample(B)
    coefs = get_coefs(STATE)
    e = np.random.randn(B)
    t = np.random.randint(STATE["T"], size=B)
    xt = coefs["C"][t] * x0 + coefs["D"][t] * e
    features = np.column_stack([xt, (t + 1.0) / STATE["T"]])
    targets = np.column_stack([e])
    return features, targets


def add_backward_paths(
    C: np.array,
    edges: np.array,
    coefs: dict,
    X0: np.array,
    model: LSRFF,
    noise_multiplier: float = 1.0,
) -> None:
    T = len(coefs["sigma"])
    E = len(edges)
    assert C.shape[1] == T
    assert C.shape[0] == E - 1
    M = len(X0)  # number of samples
    X = np.copy(X0)
    for t in reversed(range(T)):
        ephat = model.eval(np.column_stack([X, np.tile((t + 1.0) / T, (M,))])).flatten()
        muhat = (coefs["A"][t] / coefs["C"][t]) * (X - coefs["D"][t] * ephat) + coefs[
            "B"
        ][t] * X
        X = muhat + noise_multiplier * coefs["sigma"][t] * np.random.randn(M)
        counts_, edges_ = np.histogram(X, bins=edges)
        assert np.all(edges_ == edges)
        C[:, T - 1 - t] = C[:, T - 1 - t] + counts_


def create_forward_tab(session_id):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="forward-histogram-map"),
                    html.H6(
                        "Number of (forward) histogram bins (steps=%i)" % (STATE["T"])
                    ),
                    dcc.Slider(30, 200, 10, value=STATE["B"], id="forward-bins-slider"),
                    html.Button("Add 100x paths", id="forward-add-100x-button"),
                    html.Button("Add 1000x paths", id="forward-add-1000x-button"),
                    html.Button("Add 10000x paths", id="forward-add-10000x-button"),
                    html.Button("Reset", id="forward-reset-button"),
                ],
                style={"width": "67%", "display": "inline-block"},
            ),
            html.Div(
                [dcc.Graph(id="forward-histogram-slice")],
                style={
                    "width": "33%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
        ]
    )


@callback(
    Output("forward-histogram-map", "figure"),
    Output("forward-histogram-slice", "figure"),
    Input("forward-bins-slider", "value"),
    Input("forward-add-100x-button", "n_clicks"),
    Input("forward-add-1000x-button", "n_clicks"),
    Input("forward-add-10000x-button", "n_clicks"),
    Input("forward-reset-button", "n_clicks"),
    State("session-id", "data"),
)
def update_forward_histogram(
    bins_slider, n_clicks_100, n_clicks_1000, n_clicks_10000, n_clicks_reset, session_id
):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)

    def add_frwd_paths(S: int):
        eg = get_bin_edges(STATE, name="B")
        add_forward_paths(STATE["FRWD"], eg, get_schedule(STATE), get_source_sample(S))

    if not bins_slider is None and bins_slider != STATE["B"]:
        STATE.update({"B": int(bins_slider)})

    if (
        STATE["FRWD"].shape[0] != STATE["B"] or STATE["FRWD"].shape[1] != STATE["T"]
    ) or (ctx.triggered_id == "forward-reset-button" and not n_clicks_reset is None):
        STATE["FRWD"] = np.zeros((STATE["B"], STATE["T"]))

    if ctx.triggered_id == "forward-add-100x-button" and not n_clicks_100 is None:
        add_frwd_paths(100)

    elif ctx.triggered_id == "forward-add-1000x-button" and not n_clicks_1000 is None:
        add_frwd_paths(1000)

    elif ctx.triggered_id == "forward-add-10000x-button" and not n_clicks_10000 is None:
        add_frwd_paths(10000)

    set_session_cache(session_id, STATE_CACHE_VARIABLE_NAME, STATE)

    yg = get_bin_centers(STATE, name="B")
    xg = np.arange(1, STATE["T"] + 1)
    fig = px.imshow(
        STATE["FRWD"],
        title="Forward sampling histogram (%i)" % (int(np.sum(STATE["FRWD"][:, 0]))),
        origin="lower",
        x=xg,
        y=yg,
        aspect="auto",
    )
    fig.update_layout(xaxis_title="diffusion step", yaxis_title="data dimension")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=yg, y=STATE["FRWD"][:, 0], mode="lines", name="initial step")
    )
    fig2.add_trace(
        go.Scatter(
            x=yg, y=STATE["FRWD"][:, STATE["T"] // 2], mode="lines", name="midway step"
        )
    )
    fig2.add_trace(
        go.Scatter(x=yg, y=STATE["FRWD"][:, -1], mode="lines", name="final step")
    )
    fig2.update_layout(
        xaxis_title="data dimension", yaxis_title="counts", title="Forward slices"
    )

    return fig, fig2


def create_schedule_tab(session_id):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="beta-schedule"),
                    html.H6("log2(beta) range"),
                    dcc.RangeSlider(
                        -20,
                        -1,
                        1,
                        count=1,
                        value=[STATE["log2-beta-min"], STATE["log2-beta-max"]],
                        id="beta-range",
                    ),
                ],
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div(
                [
                    dcc.Graph(id="schedule-coefs"),
                    html.H6("Number of diffusion steps"),
                    dcc.Slider(50, 300, 25, value=STATE["T"], id="steps-slider"),
                ],
                style={"width": "49%", "display": "inline-block"},
            ),
            dcc.RadioItems(
                ["linear", "loglinear", "phase", "logphase", "sigma"],
                STATE["schedule"],
                inline=True,
                id="beta-schedule-type",
            ),
            dcc.Markdown(
                """
#### Notation/parametrization
- Forward diffusion: $x_t = \lambda_t x_{t-1} + \sigma_t\epsilon_t$ where $\epsilon_t\sim\mathrm{N}(0,1)$
- $\sigma_t = \sqrt{\\beta_t}$, $\lambda_t = \sqrt{1-\\beta_t}$
- Direct forward sample: $x_t|x_0 \sim \mathrm{N}(C_t x_0, D_t^2)$ 
- $C_t = \prod_{s=1}^t\lambda_s$, $D_t^2 = 1 - C_t^2$
- Conditional backward mean: $\mu_{t-1|x_0,x_t} = A_t x_0 + B_t x_t$
- $A_t=(1-\lambda_t^2)C_{t-1}/(1-C_t^2)$, $B_t=\lambda_t D_{t-1}^2 / (1-C_t^2)$
""",
                mathjax=True,
            ),
        ]
    )


@callback(
    Output("beta-schedule", "figure"),
    Input("beta-range", "value"),
    Input("steps-slider", "value"),
    Input("beta-schedule-type", "value"),
    State("session-id", "data"),
)
def update_beta_graph(beta_range, steps_slider, beta_type, session_id):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)
    STATE.update(
        {
            "T": int(steps_slider),
            "log2-beta-min": beta_range[0],
            "log2-beta-max": beta_range[1],
            "schedule": beta_type,
        }
    )
    set_session_cache(session_id, STATE_CACHE_VARIABLE_NAME, STATE)
    return px.line(
        x=np.arange(1, STATE["T"] + 1), y=get_schedule(STATE), title="beta schedule"
    ).update_layout(xaxis_title="step", yaxis_title="beta")


@callback(
    Output("schedule-coefs", "figure"),
    Input("beta-range", "value"),
    Input("steps-slider", "value"),
    Input("beta-schedule-type", "value"),
    State("session-id", "data"),
)
def update_coefs_graph(beta_range, steps_slider, beta_type, session_id):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)
    STATE.update(
        {
            "T": int(steps_slider),
            "log2-beta-min": beta_range[0],
            "log2-beta-max": beta_range[1],
            "schedule": beta_type,
        }
    )
    set_session_cache(session_id, STATE_CACHE_VARIABLE_NAME, STATE)
    steps = np.arange(1, STATE["T"] + 1)
    coefs = get_coefs(STATE)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=coefs["lambda"], mode="lines", name="lambda"))
    fig.add_trace(go.Scatter(x=steps, y=coefs["sigma"], mode="lines", name="sigma"))
    fig.add_trace(go.Scatter(x=steps, y=coefs["C"], mode="lines", name="C"))
    fig.add_trace(go.Scatter(x=steps, y=coefs["D"], mode="lines", name="D"))
    fig.add_trace(go.Scatter(x=steps, y=coefs["A"], mode="lines", name="A"))
    fig.add_trace(go.Scatter(x=steps, y=coefs["B"], mode="lines", name="B"))

    return fig.update_layout(
        xaxis_title="step", yaxis_title="coefficient", title="Step parameters"
    )


def create_fitting_tab(session_id):
    STATE = get_session_cache(session_id, "state")
    return html.Div(
        [
            html.Div(
                [
                    html.H6("Number of RFFs"),
                    dcc.Slider(
                        100,
                        1200,
                        100,
                        value=STATE["Model"].units,
                        id="rff-dimension-slider",
                    ),
                    html.H6("RFF sigma (x)"),
                    dcc.Slider(
                        -3,
                        3,
                        0.1,
                        value=np.log2(STATE["Model"].sigma[0]),
                        id="rff-sigmax-slider",
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.H6("RFF sigma (t/T)"),
                    dcc.Slider(
                        -3,
                        3,
                        0.1,
                        value=np.log2(STATE["Model"].sigma[1]),
                        id="rff-sigmat-slider",
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.H6("RFF L2 regularization"),
                    dcc.Slider(
                        -10,
                        0,
                        0.1,
                        value=np.log10(STATE["Model"].gamma),
                        id="rff-gamma-slider",
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dcc.Markdown(
                        """
#### Random Fourier Features model
- Featurization of $(x,t/T)$ is defined by length-scales $\sigma_x$, $\sigma_{t/T}$.
- Sample $M/2$ frequencies (each) $\omega_x\sim\mathcal{N}(0,\sigma_x^2)$, $\omega_{t/T}\sim\mathcal{N}(0,\sigma_{t/T}^2)$ 
- Define $M/2$ phases $\phi = \omega_x x + \omega_{t/T} t/T$
- Define $M/2$ feature tuples $(\cos\phi, \sin\phi)$, stacking them into a $M$-dim. vector $\mathbf{h}(x,t/T)$.
- The RFF model is now linear in $\mathbf{h}$: $\widehat{y}=\mathbf{h}^T\\beta$
""",
                        mathjax=True,
                    ),
                ],
                style={
                    "width": "45%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="rff-training"),
                    html.Button("Add batch (1k)", id="rff-batch-1k-button"),
                    html.Button("Add batch (5k)", id="rff-batch-5k-button"),
                    html.Button("Add batch (10k)", id="rff-batch-10k-button"),
                    html.Button("Reset", id="rff-reset-button"),
                ],
                style={
                    "width": "54%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
        ]
    )


@callback(
    Output("rff-reset-button", "n_clicks"),
    Input("rff-dimension-slider", "value"),
    Input("rff-sigmax-slider", "value"),
    Input("rff-sigmat-slider", "value"),
    Input("rff-gamma-slider", "value"),
    State("rff-reset-button", "n_clicks"),
    State("session-id", "data"),
)
def poke_reset_button(units, sigmax, sigmat, gamma, n_clicks_reset, session_id):
    n_clicks = 0 if n_clicks_reset is None else n_clicks_reset
    assert isinstance(n_clicks, int)
    # There is no need to update if the model already has the same values; then return no_update
    # This prevents the model to be reset when the fitting tab is re-entered with no real change
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)
    if (
        STATE["Model"].units == units
        and sigmax == np.log2(STATE["Model"].sigma[0])
        and sigmat == np.log2(STATE["Model"].sigma[1])
        and gamma == np.log10(STATE["Model"].gamma)
    ):
        return no_update
    return n_clicks + 1


@callback(
    Output("rff-training", "figure"),
    Input("rff-reset-button", "n_clicks"),
    Input("rff-batch-1k-button", "n_clicks"),
    Input("rff-batch-5k-button", "n_clicks"),
    Input("rff-batch-10k-button", "n_clicks"),
    State("rff-dimension-slider", "value"),
    State("rff-sigmax-slider", "value"),
    State("rff-sigmat-slider", "value"),
    State("rff-gamma-slider", "value"),
    State("session-id", "data"),
)
def handle_fitting_buttons(
    n_clicks_reset,
    n_clicks_1k,
    n_clicks_5k,
    n_clicks_10k,
    units,
    sigmax,
    sigmat,
    gamma,
    session_id,
):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)

    def update_rff_model(batchsize: int):
        X, Y = sample_ddpm_batch(STATE, batchsize)
        STATE["Model"].add_batch(X, Y)
        STATE["Model"].fit()
        Yhat = STATE["Model"].eval(X)
        RMSE = np.sqrt(np.mean(np.sum(Yhat - Y) ** 2))
        set_session_cache(session_id, STATE_CACHE_VARIABLE_NAME, STATE)

    def reset_rff_model():
        STATE.update({"Model": LSRFF(units, 2, 1, np.array([2**sigmax, 2**sigmat]))})
        STATE["Model"].set_regularization(10**gamma)
        set_session_cache(session_id, STATE_CACHE_VARIABLE_NAME, STATE)

    def make_fitting_figure():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(STATE["Model"].units),
                y=np.copy(STATE["Model"].coefs[:, 0]),
                mode="lines",
            )
        )
        fig.update_layout(
            xaxis_title="coefficient index",
            yaxis_title="coefficient value",
            title="Samples=%i" % (len(STATE["Model"])),
        )
        return fig

    if ctx.triggered_id is None:
        return make_fitting_figure()
    if ctx.triggered_id == "rff-reset-button":
        assert isinstance(n_clicks_reset, int)
        reset_rff_model()
        return make_fitting_figure()
    elif ctx.triggered_id == "rff-batch-1k-button" and not n_clicks_1k is None:
        update_rff_model(1000)
        return make_fitting_figure()
    elif ctx.triggered_id == "rff-batch-5k-button" and not n_clicks_5k is None:
        update_rff_model(5000)
        return make_fitting_figure()
    elif ctx.triggered_id == "rff-batch-10k-button" and not n_clicks_10k is None:
        update_rff_model(10000)
        return make_fitting_figure()
    else:
        print(ctx.triggered_id)
        return no_update


def create_backward_tab(session_id):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id="backward-histogram-map"),
                    html.H6(
                        "Number of (backward) histogram bins (steps=%i)" % (STATE["T"])
                    ),
                    dcc.Slider(
                        30, 200, 10, value=STATE["BB"], id="backward-bins-slider"
                    ),
                    html.Div(
                        [
                            html.H6("Noise-level: ", style={"display": "inline-block"}),
                            dcc.Input(
                                id="backward-noise-level",
                                type="number",
                                min=0.0,
                                max=2.0,
                                step=0.05,
                                value=STATE["bkwd-noise-level"],
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                    html.Button("Add 100x paths", id="backward-add-100x-button"),
                    html.Button("Add 1000x paths", id="backward-add-1000x-button"),
                    html.Button("Add 10000x paths", id="backward-add-10000x-button"),
                    html.Button("Reset", id="backward-reset-button"),
                ],
                style={"width": "67%", "display": "inline-block"},
            ),
            html.Div(
                [dcc.Graph(id="backward-histogram-slice")],
                style={
                    "width": "33%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
        ]
    )


@callback(
    Output("backward-histogram-map", "figure"),
    Output("backward-histogram-slice", "figure"),
    Input("backward-bins-slider", "value"),
    Input("backward-add-100x-button", "n_clicks"),
    Input("backward-add-1000x-button", "n_clicks"),
    Input("backward-add-10000x-button", "n_clicks"),
    Input("backward-reset-button", "n_clicks"),
    Input("backward-noise-level", "value"),
    State("session-id", "data"),
)
def update_backward_histogram(
    bins_slider,
    n_clicks_100,
    n_clicks_1000,
    n_clicks_10000,
    n_clicks_reset,
    noise_level,
    session_id,
):
    STATE = get_session_cache(session_id, STATE_CACHE_VARIABLE_NAME)

    def add_samples(S: int):
        eg = get_bin_edges(STATE, name="BB")
        add_backward_paths(
            STATE["BKWD"],
            eg,
            get_coefs(STATE),
            np.random.randn(S),
            STATE["Model"],
            noise_multiplier=STATE["bkwd-noise-level"],
        )

    if not bins_slider is None and bins_slider != STATE["BB"]:
        STATE.update({"BB": int(bins_slider)})

    if not noise_level is None and noise_level != STATE["bkwd-noise-level"]:
        STATE.update({"bkwd-noise-level": float(noise_level)})

    if (
        STATE["BKWD"].shape[0] != STATE["BB"] or STATE["BKWD"].shape[1] != STATE["T"]
    ) or (ctx.triggered_id == "backward-reset-button" and not n_clicks_reset is None):
        STATE["BKWD"] = np.zeros((STATE["BB"], STATE["T"]))

    if ctx.triggered_id == "backward-add-100x-button" and not n_clicks_100 is None:
        add_samples(100)

    elif ctx.triggered_id == "backward-add-1000x-button" and not n_clicks_1000 is None:
        add_samples(1000)

    elif (
        ctx.triggered_id == "backward-add-10000x-button" and not n_clicks_10000 is None
    ):
        add_samples(10000)

    set_session_cache(session_id, STATE_CACHE_VARIABLE_NAME, STATE)

    yg = get_bin_centers(STATE, name="BB")
    xg = np.arange(1, STATE["T"] + 1)
    fig = px.imshow(
        STATE["BKWD"],
        title="Backward sampling histogram (%i)" % (int(np.sum(STATE["BKWD"][:, 0]))),
        origin="lower",
        x=xg,
        y=yg,
        aspect="auto",
    )
    fig.update_layout(
        xaxis_title="reverse diffusion step", yaxis_title="data dimension"
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=yg, y=STATE["BKWD"][:, 0], mode="lines", name="initial step")
    )
    fig2.add_trace(
        go.Scatter(
            x=yg, y=STATE["BKWD"][:, STATE["T"] // 2], mode="lines", name="midway step"
        )
    )
    fig2.add_trace(
        go.Scatter(x=yg, y=STATE["BKWD"][:, -1], mode="lines", name="final step")
    )
    fig2.update_layout(
        xaxis_title="data dimension", yaxis_title="counts", title="Backward slices"
    )

    return fig, fig2


def serve_layout():
    session_id = str(uuid.uuid4())

    set_session_cache(
        session_id, STATE_CACHE_VARIABLE_NAME, create_default_sesh_state()
    )

    return html.Div(
        [
            dcc.Store(id="session-id", data=session_id),
            html.H3("Denoising Diffusion Probabilistic Model (DDPM) demonstration"),
            dcc.Tabs(
                id="tabs",
                value="tab-frwd",
                children=[
                    dcc.Tab(label="Noise Schedule", value="tab-schedule"),
                    dcc.Tab(label="Forward Diffusion", value="tab-frwd"),
                    dcc.Tab(label="LSQ RFF Fitting", value="tab-fitting"),
                    dcc.Tab(label="Backward Diffusion", value="tab-bkwd"),
                ],
            ),
            html.Div(id="tabs-content"),
        ]
    )


app.layout = serve_layout


@callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    State("session-id", "data"),
)
def render_content(tab, sesh):
    if tab == "tab-frwd":
        return html.Div([create_forward_tab(sesh)])
    elif tab == "tab-fitting":
        return html.Div([create_fitting_tab(sesh)])
    elif tab == "tab-schedule":
        return html.Div([create_schedule_tab(sesh)])
    elif tab == "tab-bkwd":
        return html.Div([create_backward_tab(sesh)])


if __name__ == "__main__":
    app.run_server(debug=True, threaded=False, use_reloader=False)
