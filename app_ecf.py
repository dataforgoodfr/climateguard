from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from st_aggrid import AgGrid, JsCode

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
    df = df.loc[
        (pd.to_datetime(df.start) >= datetime.strptime("2023-09-01", "%Y-%m-%d"))
        & (pd.to_datetime(df.start) < datetime.strptime("2024-09-01", "%Y-%m-%d"))
    ]
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

    # Mega cheat for today
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "id": "0eb5789fa23e0819f817ea10fe1fccd19e61e40a1239cc93f701fd56bd8ea66f",
                        "text": "un discours radical qui met des spécialistes mal à l' aise parce qu' en fait il dit au fond est dit n' importe quoi il le dit avec beaucoup de talent sans doute mais eddy juste n' importe quoi jean-marc jancovici quand même euh euh est quelqu' un d' extrêmement intelligent qui a eu quand même une grande vertu je vais je le souligne parce qu' il faut le reconnaître l' un des premiers dans cette mouvance décroissant cette sinon le premier à avoir compris l' importance des émissions de gaz à effet de serre les écologistes n' en parlait pas du tout pas du tout dans les années quatre-vingt-dix et c' est lui qui les alertes là-dessus et euh là euh ça ce qui l' a rendu très populaire est aussi puissant c' est que il va faire la jonction entre la droite et extrême l' gauche extrême finalement gauche la finalement droite la l' droite adorent l' adorent parce que qu' il est pronucléaire il est pronucléaire parce qu' il est tellement convaincu que l' humanité court à sa perte mais se dit on va prendre tous les outils pour permettre d' atténuer la chute donc promet pronucléaire mais aussi fondamentalement décroissant et ça s' appelait à l' extrême gauche et à la gauche radicale euh on a décidé de lui consacrer un un portrait parce qu' il est important d' être bien conscient des sous-jacent son discours euh il est convaincu que l' humanité va s' voilà c' effondrer est voilà c' un est un c' est ce qu' on qu' dise on dise qu' on appelle un collapsologues euh et euh enfin ses névroses sont pas obligés on n' est pas obligé écouter ses evros finalement des solutions à assez radicale et qui sont assez autocratique il a dit tout le bien qui pensait d' un régime comme le régime chinois qui avait l' avantage de pouvoir imposer des mesures douloureuses à une population il pense que euh la démocratie mais comme d' ailleurs euh nombre de ces de ces militants décroissants euh je pense que la démocratie n' est simplement simplement pas un régime régime adapté euh à l' urgence de la situation l' urgence de la situation imposerait la confiscation du pouvoir par une caste de scientifiques qui serait mieux que le bon peuple ce qu' il convient de faire pour sauver la planète c' est pas rien quand même et ses jambes mais la planète ça peut juste oui non mais c' est ici ces gens n' aiment pas le peuple depuis qu' en souvenir des classes populaires qu' on a du mal à boucler les fins de mois qu' on est ouvriers employés franchement l' écologie c' est pas une priorité et je pense qu' il faut d' abord pensé aux gens faire en sorte qu' ils vivent d' abord pensé aux gens faire en sorte qu' ils vivent bien et ensuite avec un vrai mouvement collectif de fonds basé sur la science on s' occupera de ce qu' est ce qu' on est fan de moi quand le plus souvent confrontés aux phénomènes les plus durs concernant le réchauffement climatique non mais le losc ce que le réchauffement climatique est vraiment un sujet fondamental oui mais on a les moyens mais on a les moyens et le devoir d' ailleurs en tant que puissance économique important d' de eux prendre de ce prendre sujet ce à sujet à bras bras-le-corps euh sauf madame vous solid y a une folie et j' avais eu la même folie et que le covid de vouloir nous les hommes influer sur certaines choses sur lesquelles on ne peut pas influer c' est-à-dire et je suis pas sûr que l' homme puisse influer sur le climat sauf effectivement avec des mesures radicales de donner un exemple ce que vous dites est pas vrai euh les données qu' il est dix heures trente vous avez donné l' exemple mais avez il donné y l' exemple a mais une il sorte y a de une folie sorte de folie de penser que l' homme et puis je vais vous donner un exemple et j' ai vu avec les inondations par exemple de valence c' est très intéressant en mille neuf cent cinquante-sept ia humide mort je crois que c' était il y avait eu exactement là elle est exactement le même c' est pas la même fréquence sept b vous en savez rien puisque cette arrivée arrive demain la même chose je vous donnerai mon mais quand je me c' est vrai que je suis très je me méfie beaucoup je ne peux pas vous dire autre chose parfois dire c' autre est juste chose parfois c' est pas juste juste parfois donc je c' prends est pas juste donc je prends effectivement un peu de distance sur toutes ces infos parce que je sais derrière le discours euh ce que ça peut impliquer et les motivations",
                        "start": pd.Timestamp("2024-11-07 13:27:00"),
                        "channel_name": "itele",
                        "channel_is_radio": False,
                        "channel_program_type": "Information continue",
                        "channel_program": "Information continue",
                        "claims": [
                            {
                                "claim": "L'humanité peut influencer le climat seulement par des mesures radicales.",
                                "context": "Un intervenant évoque le point de vue selon lequel il n'est pas certain que l'homme puisse influer sur le climat et que cela nécessiterait des mesures radicales.",
                                "analysis": "Cette allégation simplifie la complexité des causes du changement climatique et minimise la reconnaissance du consensus scientifique sur l'influence humaine sur le climat. Elle pourrait semer le doute sur les capacités d'action des sociétés modernes face à la crise climatique.",
                                "disinformation_score": "high",
                                "disinformation_category": "consensus",
                                "score": "high",
                                "pro_anti": "anti-écologie",
                            },
                            {
                                "claim": "Des personnes comme Jean-Marc Jancovici pensent que la démocratie n'est pas adaptée à l'urgence climatique et prônent une confiscation du pouvoir par une élite scientifique.",
                                "context": "L'intervenant critique le discours de Jancovici, l'accusant de promouvoir des idées autocratiques face à la crise climatique.",
                                "analysis": "Cette déclaration pourrait déformer la position de Jancovici en exagérant son approche politique et ses implications. Cela nécessite une vérification pour comprendre la nuance de son discours et éviter de colporter des interprétations erronées.",
                                "disinformation_score": "medium",
                                "disinformation_category": "narrative",
                                "score": "medium",
                                "pro_anti": "anti-écologie",
                            },
                            {
                                "claim": "Il y a eu des inondations en 1957 qui sont citées comme des exemples d'événements météorologiques similaires à ceux d'aujourd'hui, insinuant que le changement climatique n'est pas nécessairement lié aux actions humaines.",
                                "context": "L'intervenant cite une inondation à Valence en 1957 pour argumenter que les catastrophes climatiques se produisent indépendamment de l'impact humain, tout en minimisant le changement climatique actuel.",
                                "analysis": "Bien que les inondations soient un phénomène naturel, utiliser un événement historique pour minimiser le lien entre changements climatiques actuels et activité humaine peut être trompeur. Cela nécessite une vérification approfondie des données sur les tendances climatiques.",
                                "disinformation_score": "medium",
                                "disinformation_category": "facts",
                                "score": "medium",
                                "pro_anti": "anti-écologie",
                            },
                        ],
                        "max_claim_severity": "high",
                        "num_claims": 3,
                    }
                ]
            ),
        ],
        axis=0,
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
                marker_colors=["#e38690", "#ead98b"],
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
        legend=dict(font=dict(color="dark grey")),
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
    df["month"] = pd.to_datetime(df.start).dt.strftime("%b %Y")
    # Group by month and channel_name to get claim counts
    df_grouped = df[["month", "channel_name", "num_claims"]]
    df_grouped["high_risk_claims"] = (df_grouped.num_claims == 5).astype(int)
    monthly_claims = (
        df_grouped.groupby(["month", "channel_name"])["high_risk_claims"]
        .sum()
        .reset_index()
    )
    monthly_claims = monthly_claims.sort_values(
        by="month", key=lambda x: pd.to_datetime(x, format="%b %Y")
    )
    print(monthly_claims.month.unique())

    # Define a colormap for the channels
    colormap = ["#9dc6d8", "#00b3ca", "#7dd0d6", "#1d4e89"]

    # Create stacked bar chart
    fig = go.Figure()

    # Add bars for each channel
    for idx, channel in enumerate(monthly_claims["channel_name"].unique()):
        channel_data = monthly_claims[monthly_claims["channel_name"] == channel]
        fig.add_trace(
            go.Bar(
                name=channel,
                x=channel_data["month"],
                y=channel_data["high_risk_claims"],
                hovertemplate="Month: %{x}<br>Claims: %{y}<br>Channel: "
                + channel
                + "<extra></extra>",
                marker_color=colormap[idx],  # Default color if not specified
            )
        )

    # Update layout
    fig.update_layout(
        barmode="stack",
        title="Number of Detected Claims Over Time",
        xaxis_title="Month",
        yaxis_title="Number of Claims",
        width=900,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="dark grey")),
        xaxis=dict(
            tickmode="array",
            ticktext=monthly_claims.month.unique(),
            tickvals=list(range(1, len(monthly_claims.month.unique()) + 1)),
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="dark grey"),
            color="dark grey",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)",
            tickfont=dict(color="dark grey"),
            color="dark grey",
            tickformat=",d",  # Format ticks as integers
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
                "sort": "desc",
                "headerName": "Diffusion date",
                "type": ["dateColumnFilter", "shortDateTimeFormat"],
            },
            {"field": "text", "headerName": "Transcript", "flexgrow": 1},
            {
                "field": "max_claim_severity",
                "width": 50,
                # "sort": "desc",
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
    average_score = round(max(scores), 1)

    # st.metric("Disinformation risk score", f"{average_score}/5")

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
df = load_detected_claims()
show_kpis(df)
st.plotly_chart(generate_stacked_bar_chart(df), use_container_width=True)

selected_transcript = show_transcripts_and_select_one(df)
if selected_transcript:
    show_transcript_details(selected_transcript)
