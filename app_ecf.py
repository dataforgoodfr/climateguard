import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from st_aggrid import AgGrid, JsCode

from live_detect import live_detect

load_dotenv()


def set_page_config():
    # Set page configuration
    st.set_page_config(
        page_title="CLIMATEGUARD: CLIMATE DISINFORMATION ANALYSIS ON TV TRANSMISSIONS",
        layout="wide",
    )
    # Set font to Poppins
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
        html, body, h1, h2, h3, h4, h5, h6, p, div, span, button, label, input, select, textarea {
            font-family: 'Poppins', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Main app
    st.title("CLIMATEGUARD: CLIMATE DISINFORMATION ANALYSIS")


@st.cache_data
def load_detected_claims() -> pd.DataFrame:
    df = pd.read_parquet(
        "data/raw/4_channels_predictions_09_2023_09_2024.parquet"
    )  # .sample(100)
    # Keep only anti-climate claims
    df["claims"] = df["claims"].apply(
        lambda claims: [
            claim for claim in claims if "anti-écologie" in claim["pro_anti"].lower()
        ]
    )
    print(df.columns)
    df["max_claim_severity"] = df["claims"].apply(
        lambda claims: max(
            [
                SCORES_MAPPING.get(claim["disinformation_score"], 0)
                if "anti-écologie" in claim["pro_anti"].lower()
                and isinstance(SCORES_MAPPING, dict)
                else 0
                for claim in claims
            ]
        )
        if len(claims)
        else 0
    )

    # Only show transcripts for which we detected at least one claim
    df["num_claims"] = df["claims"].str.len()

    return df


class Claim(BaseModel):
    claim: str
    context: str
    analysis: str
    disinformation_score: str
    disinformation_category: str
    pro_anti: str


class Claims(BaseModel):
    claims: list[Claim]


class Transcript(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    start: pd.Timestamp
    text: str
    channel_name: str
    channel_is_radio: bool
    channel_program_type: str
    channel_program: str
    claims: list[Claim]


def generate_pie_chart(df, title):
    # Calculate the counts
    high_risk_count = (df["max_claim_severity"] == 5).sum()
    other_count = (df["max_claim_severity"] != 5).sum()

    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["High Risk Claims", "Other Claims"],
                values=[high_risk_count, other_count],
                marker_colors=["#ff5e00", "#00c8ff"],
                textinfo="percent",  # Show both percentage and label
                textposition="inside",  # Position labels outside the pie
                texttemplate="<b>%{percent:.1%}</b>",  # Format with 1 decimal place and bold
                hovertemplate="<br>%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={"text": title, "x": 0.4, "xanchor": "center"},
        showlegend=True,
        width=700,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color="white")),
    )
    return fig


def get_percentage_of_high_risk_claims(df: pd.DataFrame) -> float:
    high_risk_count = (df["max_claim_severity"] == 5).sum()
    other_count = (df["max_claim_severity"] != 5).sum()
    return high_risk_count / (high_risk_count + other_count)


def show_kpis(df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3, gap="large")

    # Add pie chart to col1
    with col1:
        fig = generate_pie_chart(
            df, title="Percentage of extracts with <br>high disinformation risk claims"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")

        st.markdown(
            """
            <div style="display: flex; flex-direction: column; justify-content: center;">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""From Date:  
            
            {pd.to_datetime(df.start.min()).strftime("%d/%m/%Y")}
""",
        )
        st.markdown(
            f"""To Date:  
            
            {pd.to_datetime(df.start.max()).strftime("%d/%m/%Y")}
""",
        )
        # Calculate number of months using to_period
        num_months = (
            len(
                pd.period_range(
                    start=pd.to_datetime(df.start.min()),
                    end=pd.to_datetime(df.start.max()),
                    freq="M",
                )
            )
            + 1
        )
        st.markdown(
            f"""Number of months of analyzed data:  
            
            {num_months}""",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Center metrics vertically in col4
    with col3:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; justify-content: center;">
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""Number of transcripts analyzed:  
            
            {len(df)}
""",
        )
        st.markdown(
            f"""Percentage of transcripts with high disinformation risk claims:  
            
            {get_percentage_of_high_risk_claims(df):.2%}
""",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def generate_stacked_bar_chart(df):
    df["month"] = pd.to_datetime(df.start).dt.month
    # Group by month and channel_name to get claim counts
    df_grouped = df[["month", "channel_name", "num_claims"]]
    df_grouped["high_risk_claims"] = (df_grouped.num_claims == 5).astype(int)
    monthly_claims = (
        df_grouped.groupby(["month", "channel_name"])["high_risk_claims"]
        .sum()
        .reset_index()
    )

    # Create stacked bar chart
    fig = go.Figure()

    # Add bars for each channel
    for channel in monthly_claims["channel_name"].unique():
        channel_data = monthly_claims[monthly_claims["channel_name"] == channel]
        fig.add_trace(
            go.Bar(
                name=channel,
                x=channel_data["month"],
                y=channel_data["high_risk_claims"],
                hovertemplate="Month: %{x}<br>Claims: %{y}<br>Channel: "
                + channel
                + "<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        barmode="stack",
        title=None,
        xaxis_title="Month",
        yaxis_title="Number of Claims",
        width=900,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="white")),
        xaxis=dict(
            tickmode="array",
            ticktext=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
            tickvals=list(range(1, 13)),
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="white"),
            color="white",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="white"),
            color="white",
        ),
    )
    fig.update_layout(
        dict(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=["visible", "legendonly"],
                                label="Deselect All",
                                method="restyle",
                            ),
                            dict(
                                args=["visible", True],
                                label="Select All",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=1,
                    xanchor="right",
                    y=1.1,
                    yanchor="top",
                ),
            ]
        )
    )

    return fig


def show_transcripts_and_select_one(df: pd.DataFrame) -> Transcript:
    st.write("## Transcripts with claims")

    df["max_claim_severity"] = df["claims"].apply(
        lambda claims: max(
            [
                SCORES_MAPPING.get(claim["disinformation_score"], 0)
                if claim["pro_anti"] == "anti-écologie"
                and isinstance(SCORES_MAPPING, dict)
                else 0
                for claim in claims
            ]
        )
        if len(claims)
        else 0
    )

    # Only show transcripts for which we detected at least one claim
    df["num_claims"] = df["claims"].str.len()
    df = df[(df["num_claims"] > 0) & (df["max_claim_severity"] > 0)]

    table = df.reset_index()[
        ["id", "channel_name", "start", "text", "num_claims", "max_claim_severity"]
    ]

    # https://www.ag-grid.com/javascript-data-grid/grid-options/
    grid_options = {
        "columnDefs": [
            {
                "field": "channel_name",
                "headerName": "Channel",
                "width": 50,
                "suppressMenu": False,
                "enableRowGroup": False,
                "rowGroup": False,
            },
            {
                "field": "start",
                "width": 50,
                "headerName": "Diffusion date",
                "type": ["dateColumnFilter", "shortDateTimeFormat"],
            },
            {"field": "text", "headerName": "Transcript", "flexgrow": 1},
            {
                "field": "max_claim_severity",
                "width": 50,
                "sort": "desc",
                "headerName": "Max claim severity",
                "valueFormatter": JsCode("""
function(params) {
  if (!params.value){return "";}
  return params.value + "/5";}
"""),
            },
            {
                "field": "num_claims",
                "width": 50,
                "headerName": "Number of claims",
                "valueFormatter": JsCode("""
function(params) {
  if (!params.value){return "";}
  return params.value;}
"""),
            },
        ],
        "autoSizeStrategy": {"type": "fitCellContents", "skipHeader": False},
        "defaultColDef": {"filter": True, "suppressMenu": True},
        "rowSelection": "multiple",
        "onColumnVisible": JsCode("function(params){params.api.sizeColumnsToFit();}"),
        "onGridColumnsChanged": JsCode(
            "function(params){params.api.sizeColumnsToFit();}"
        ),
        "onToolPanelVisibleChanged": JsCode(
            "function(params){params.api.sizeColumnsToFit();}"
        ),
        "rowBuffer": 50,
        "sideBar": {
            "toolPanels": [
                {
                    "id": "columns",
                    "labelDefault": "Columns",
                    "labelKey": "columns",
                    "iconKey": "columns",
                    "toolPanel": "agColumnsToolPanel",
                    "toolPanelParams": {
                        # https://www.ag-grid.com/javascript-data-grid/tool-panel-columns/#section-visibility
                        "contractColumnSelection": False,
                        "suppressColumnFilter": True,
                        "suppressPivotMode": True,
                        "suppressRowGroups": False,
                        "suppressValues": True,
                    },
                }
            ],
        },
        "groupDefaultExpanded": 1,
    }
    grid_response = AgGrid(
        table,
        height=400,  # Initial height
        gridOptions=grid_options,
        fit_columns_on_grid_load=True,
        filter=True,
        allow_unsafe_jscode=True,
    )
    try:
        selected_transcript_iloc = next(int(i) for i in grid_response.selected_rows_id)
        selected_transcript = df.iloc[selected_transcript_iloc]
        return Transcript(**selected_transcript.to_dict())
    except Exception as e:
        print(e)
        st.info("Select one transcript to explore its analysis.", icon="ℹ️")
        # st.stop()


def show_transcript_details(transcript: Transcript) -> None:
    st.write("## Transcript analysis")
    st.markdown(f"""
- Channel: **{transcript.channel_name}**
- Channel is radio: **{transcript.channel_is_radio}**
- Program type: **{transcript.channel_program_type}**
- Program: **{transcript.channel_program.strip()}**
- Diffusion date: **{transcript.start}**
""")
    st.chat_message("human").write(transcript.text)
    show_claims_details(transcript.claims)


def show_claims_details(claims: list[Claim]) -> None:
    # Sort claims by high/low severity
    claims = sorted(claims, key=lambda c: -SCORES_MAPPING[c.disinformation_score])

    scores = [SCORES_MAPPING[c.disinformation_score] for c in claims]
    average_score = round(sum(scores) / len(scores), 1)

    st.metric("Disinformation risk average score", f"{average_score}/5")

    for claim in claims:
        score = SCORES_MAPPING[claim.disinformation_score]
        col1, col2 = st.columns([1, 1])
        with col1:
            if score == 0:
                st.success(f"### {claim.claim}\n{claim.context}")
            elif score == 1:
                st.info(f"### {claim.claim}\n{claim.context}")
            elif score == 2:
                st.warning(f"### {claim.claim}\n{claim.context}")
            else:
                st.error(f"### {claim.claim}\n{claim.context}")

        with col2:
            st.metric(
                "Disinformation risk score",
                f"{claim.disinformation_score} ({score}/5)",
            )
            st.markdown(f"{claim.analysis}")


set_page_config()

tab0, tab1 = st.tabs(["Transcripts analysis", "Live claims detection"])


with tab0:
    SCORES_MAPPING = {
        "very low": 0,
        "low": 1,
        "medium": 2,
        "high": 5,
    }
    df = load_detected_claims()
    show_kpis(df)
    st.plotly_chart(generate_stacked_bar_chart(df), use_container_width=True)

    selected_transcript = show_transcripts_and_select_one(df)
    if selected_transcript:
        show_transcript_details(selected_transcript)
with tab1:
    live_detect()
