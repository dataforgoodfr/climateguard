import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from st_aggrid import AgGrid, JsCode

load_dotenv()


def set_page_config():
    # Set page configuration
    st.set_page_config(page_title="CLIMATE DISINFORMATION", layout="wide")
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
    st.title("CLIMATE DISINFORMATION DETECTION")


@st.cache_data
def load_deteted_claims() -> pd.DataFrame:
    df = pd.read_parquet(
        "data/raw/3_channels_predictions_09_2023_09_2024.parquet"
    )  # .sample(100)
    # Keep only anti-climate claims
    df["claims"] = df["claims"].apply(
        lambda claims: [
            claim for claim in claims if claim["pro_anti"] == "anti-écologie"
        ]
    )
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


def show_kpis(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of transcripts analyzed", len(df))


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
        st.stop()


def show_transcript_details(transcript: Transcript) -> None:
    st.write("## Transcript analysis")
    st.markdown(f"""
- Channel: **{transcript.channel_name}**
- Channel is radio: **{transcript.channel_is_radio}**
- Program type: **{transcript.channel_program_type}**
- Program: **{transcript.channel_program}**
- Diffusion date: **{transcript.start}**
""")
    st.chat_message("human").write(transcript.text)
    show_claims_details(transcript.claims)


def show_claims_details(claims: list[Claim]) -> None:
    # Sort claims by high/low severity
    claims = sorted(claims, key=lambda c: -SCORES_MAPPING[c.disinformation_score])

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
SCORES_MAPPING = {
    "very low": 0,
    "low": 1,
    "medium": 2,
    "high": 5,
}
df = load_deteted_claims()
show_kpis(df)
selected_transcript = show_transcripts_and_select_one(df)
show_transcript_details(selected_transcript)
