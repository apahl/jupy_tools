"""Calculate compound IC50 values from a dose-response table."""

import argparse
import base64
import html
import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import utils as u, mol_view as mv


def four_parameter_logistic(
    concentration: np.ndarray,
    bottom: float,
    top: float,
    ic50: float,
    hill_slope: float,
) -> np.ndarray:
    """Return a four-parameter logistic curve with a positive dose response."""
    return bottom + (top - bottom) / (1.0 + (ic50 / concentration) ** hill_slope)


def fit_ic50(
    concentration: np.ndarray, response: np.ndarray, direction: str = "increasing"
) -> tuple[float, str, np.ndarray]:
    """Fit one curve and return its IC50, relation, and curve parameters."""
    tested_min = float(np.min(concentration))
    tested_max = float(np.max(concentration))
    initial_ic50 = float(np.sqrt(tested_min * tested_max))
    if "inc" not in direction and "dec" not in direction:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    slope_sign = 1.0 if "inc" in direction else -1.0
    initial_guess = (
        float(np.min(response)) if slope_sign > 0 else float(np.max(response)),
        float(np.max(response)) if slope_sign > 0 else float(np.min(response)),
        initial_ic50,
        slope_sign,
    )

    parameters, _ = curve_fit(
        four_parameter_logistic,
        concentration,
        response,
        p0=initial_guess,
        bounds=(
            [-1000.0, -1000.0, 1e-12, -20.0 if slope_sign < 0 else 1e-6],
            [1000.0, 1000.0, np.inf, -1e-6 if slope_sign < 0 else 20.0],
        ),
        maxfev=100000,
    )
    fitted_ic50 = float(parameters[2])

    if fitted_ic50 < tested_min:
        relation = "<"
    elif fitted_ic50 > tested_max:
        relation = ">"
    else:
        relation = "="
    return fitted_ic50, relation, parameters


def plot_to_data_uri(
    identifier: str,
    concentration: np.ndarray,
    response: np.ndarray,
    fitted_parameters: np.ndarray | None,
    ic50: float,
    relation: str,
    direction: str,
) -> str:
    """Render one dose-response curve as an embedded PNG data URI."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.scatter(concentration, response, color="#1769aa", label="Mean inhibition")
    if fitted_parameters is not None:
        curve_x = np.geomspace(np.min(concentration), np.max(concentration), 300)
        curve_y = four_parameter_logistic(curve_x, *fitted_parameters)
        axis.plot(curve_x, curve_y, color="#d1495b", label="4PL fit")
        axis.axvline(
            ic50,
            color="#d1495b",
            linestyle="--",
            alpha=0.7,
            label=f"IC50: {ic50:.4g} uM",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Concentration (uM)")
    axis.set_ylabel("Inhibition (%)")
    axis.set_title(
        f"{identifier}  |  {direction}  |  IC50 {relation} {ic50:.4g} uM"
        if fitted_parameters is not None
        else f"{identifier}  |  fit failed"
    )
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    figure.tight_layout()

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=120)
    plt.close(figure)
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def write_html_report(
    df: pd.DataFrame, title: str, fn: Path, id_col: str, plots: list[dict[str, object]]
) -> None:
    """Write a self-contained HTML report containing all compound plots."""
    df = df.copy()
    df_plots = pd.DataFrame(plots)[["identifier", "image"]]
    df_plots["image"] = df_plots["image"].apply(
        lambda x: f'<img width="400" src="{x}" alt="Dose-response curve">'
    )
    df_plots = df_plots.rename(columns={"identifier": id_col, "image": "DRC"})
    df = pd.merge(df, df_plots, how="left", on=id_col)
    mv.write_mol_table(df, title=title, fn=fn, id_col=id_col)


# def write_html_report(
#     output_path: Path, plots: list[dict[str, object]], results: pd.DataFrame
# ) -> None:
#     """Write a self-contained HTML report containing all compound plots."""
#     sections = []
#     for plot in plots:
#         identifier = html.escape(str(plot["identifier"]))
#         status = html.escape(str(plot["status"]))
#         sections.append(
#             f"<section><h2>{identifier}</h2><p>{status}</p>"
#             f'<img src="{plot["image"]}" alt="Dose-response curve for {identifier}"></section>'
#         )
#     document = f"""<!doctype html>
# <html lang="en">
# <head>
# <meta charset="utf-8">
# <meta name="viewport" content="width=device-width, initial-scale=1">
# <title>IC50 dose-response curves</title>
# <style>
# body {{ font-family: sans-serif; margin: 2rem auto; max-width: 900px; color: #202124; }}
# header {{ border-bottom: 1px solid #ccc; margin-bottom: 2rem; }}
# section {{ border-bottom: 1px solid #ddd; padding: 1rem 0 2rem; }}
# h2 {{ margin-bottom: 0.25rem; }}
# p {{ color: #555; margin-top: 0; }}
# img {{ display: block; max-width: 100%; height: auto; }}
# </style>
# </head>
# <body>
# <header><h1>IC50 dose-response curves</h1><p>{len(results)} compounds</p></header>
# {"".join(sections)}
# </body>
# </html>
# """
#     output_path.write_text(document, encoding="utf-8")


def calc_ic50(
    df: pd.DataFrame,
    id_col: str = "Compound_Id",
    conc_col: str = "concentration",
    conc_unit: str = "uM",
    value_col: str = "value",
    direction: str = "increasing",
    create_plots: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Calculate IC50 values for each identifier in the given DataFrame."""
    data = df.copy()
    required_columns = {id_col, conc_col, value_col}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    unit_factors = {"nM": 1e-3, "uM": 1.0, "mM": 1e3}
    if conc_unit not in unit_factors:
        raise ValueError("conc_unit must be one of: nM, uM, mM")

    measurements = data[[id_col, conc_col, value_col]].copy()
    measurements[conc_col] = pd.to_numeric(measurements[conc_col], errors="coerce")
    measurements[value_col] = pd.to_numeric(measurements[value_col], errors="coerce")
    measurements = measurements.dropna(subset=[id_col, conc_col, value_col])
    measurements = measurements.rename(
        columns={
            id_col: "identifier",
            conc_col: "concentration",
            value_col: "value",
        }
    )
    measurements["concentration"] *= unit_factors[conc_unit]
    if (measurements["concentration"] <= 0).any():
        raise ValueError("All concentrations must be greater than zero")

    mean_response = (
        measurements.groupby(["identifier", "concentration"], as_index=False)["value"]
        .mean()
        .sort_values(["identifier", "concentration"])
    )

    results: list[dict[str, object]] = []
    plots: list[dict[str, object]] = []
    for identifier, curve in mean_response.groupby("identifier", sort=True):
        concentration = curve["concentration"].to_numpy(dtype=float)
        response = curve["value"].to_numpy(dtype=float)
        result: dict[str, object] = {
            id_col: identifier,
            "ic50_uM": np.nan,
            "relation": "",
            "fit_status": "ok",
            "concentrations_used": len(curve),
        }
        fitted_parameters = None
        try:
            result["ic50_uM"], result["relation"], fitted_parameters = fit_ic50(
                concentration, response, direction
            )
        except (RuntimeError, ValueError, OverflowError) as error:
            result["fit_status"] = f"failed: {error}"
        if create_plots:
            plots.append(
                {
                    "identifier": identifier,
                    "status": result["fit_status"],
                    "image": plot_to_data_uri(
                        str(identifier),
                        concentration,
                        response,
                        fitted_parameters,
                        float(result["ic50_uM"] or np.nan),
                        str(result["relation"]),
                        direction,
                    ),
                }
            )
        results.append(result)

    return pd.DataFrame(results), plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("input/export_EOS300147_138.xlsx"),
        help="Input Excel workbook.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("ic50_results.tsv"),
        help="Output TSV path (default: ic50_results.tsv).",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("ic50_curves.html"),
        help="Output HTML report path (default: ic50_curves.html).",
    )
    parser.add_argument(
        "--identifier-column", default="eos", help="Identifier column name."
    )
    parser.add_argument(
        "--concentration-column",
        default="concentration",
        help="Concentration column name.",
    )
    parser.add_argument(
        "--concentration-unit",
        choices=("nM", "uM", "mM"),
        default="uM",
        help="Unit of the concentration column (default: uM).",
    )
    parser.add_argument(
        "--value-column", default="value", help="Response value column name."
    )
    parser.add_argument(
        "--direction",
        choices=("increasing", "decreasing", "inc", "dec"),
        default="increasing",
        help="Expected response direction as concentration increases (default: increasing).",
    )
    args = parser.parse_args()

    results, plots = calc_ic50(
        args.input,
        args.id_col,
        args.conc_col,
        args.conc_unit,
        args.value_col,
        args.direction,
    )
    results.to_csv(args.output, index=False, sep="\t", float_format="%.8g")
    write_html_report(args.html, plots, results)
    print(f"Wrote {len(results)} compound results to {args.output}")
    print(f"Wrote {len(plots)} dose-response plots to {args.html}")


if __name__ == "__main__":
    main()
