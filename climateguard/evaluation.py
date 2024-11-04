from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix


def show_classification_report(y_true: Iterable[str], y_pred: Iterable[str]) -> None:
    print(classification_report(y_true, y_pred, zero_division=0))


def format_confusion_matrix_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the DataFrame according to specified rules:
    - If value == 0: return "0"
    - If value > 0 and with decimals: "{value}<br>{percentage:.2%}"
    - If value > 0 and without decimals: "{value}<br>{percentage:.0%}"
    where percentage is the value divided by the sum of the row values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to format.

    Returns:
    pd.DataFrame: A new DataFrame with formatted strings.
    """
    # Calculate the row sums
    row_sums = df.sum(axis=1)

    # Define the formatting function
    def format_value(value: float, row_sum: "pd.Series[float]") -> str:
        if value == 0:
            return "0"
        else:
            percentage = value / row_sum
            if percentage * 100 % 1 != 0:  # Check for decimals
                return f"{value:.0f}<br>{percentage:.2%}"
            else:
                return f"{value:.0f}<br>{percentage:.0%}"

    # Apply the formatting function to the DataFrame
    formatted_df: pd.DataFrame = df.apply(
        lambda row: row.apply(lambda x: format_value(x, row_sums[row.name])), axis=1
    )

    return formatted_df


def plot_confusion_matrix(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: list[str],
    target_name: list[str],
) -> None:
    # Confusion matrix array

    # We use a normalized confusion matrix for colors
    # because of the class imbalance
    cm_matrix_colors = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    # Confusion Matrix text format
    cm_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    cm_text = format_confusion_matrix_text(pd.DataFrame(cm_matrix))

    # Plot
    go.Figure(
        data=go.Heatmap(
            z=cm_matrix_colors,
            x=target_name,
            y=target_name,
            colorscale="Blues",
            showscale=False,
            text=cm_text,
            texttemplate="%{text}",
            hoverinfo="none",
        ),
        layout=go.Layout(
            title="Confusion Matrix",
            title_xref="paper",
            title_x=0.5,
            title_y=0.97,
            xaxis_title="Predicted class",
            yaxis_title="True class",
            margin=dict(r=25, t=40, b=25),
            width=600,
            height=500,
            yaxis_autorange="reversed",
            xaxis_tickangle=-30,
        ),
    ).show()


def plot_binary_confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str]) -> None:
    labels = [False, True]
    target_names = ["0_accepted", "[1-5]_contrarian"]
    plot_confusion_matrix(y_true, y_pred, labels, target_names)


def plot_cards_confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str]) -> None:
    labels = [
        "0_accepted",
        "1_its_not_happening",
        "2_its_not_us",
        "3_its_not_bad",
        "4_solutions_wont_work",
        "5_science_is_unreliable",
    ]
    target_names = labels
    plot_confusion_matrix(y_true, y_pred, labels, target_names)


def evaluation_report(y_true: Iterable[str], y_pred: Iterable[str]) -> None:
    # Binary labels
    print("BINARY CLASSIFICATION REPORT\n" + "=" * 28 + "\n")
    y_true_binary = pd.Series(y_true) != "0_accepted"
    y_pred_binary = pd.Series(y_pred) != "0_accepted"
    show_classification_report(y_true_binary, y_pred_binary)
    plot_binary_confusion_matrix(y_true_binary, y_pred_binary)

    # CARDS labels
    print("CARDS CLASSIFICATION REPORT\n" + "=" * 27 + "\n")
    show_classification_report(y_true, y_pred)
    plot_cards_confusion_matrix(y_true, y_pred)
