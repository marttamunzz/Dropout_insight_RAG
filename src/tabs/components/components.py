from io import StringIO

import dash_bootstrap_components as dbc
import dice_ml as dml
import pandas as pd
import plotly.express as px

import time
import re
from dash import dcc, html, Input, Output, State, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from explainerdashboard.custom import *

from rag.rag_recommender import generate_rag_recommendations
from rag.llm_providers import LLM_MODELS

from dice_ml.model_interfaces.base_model import BaseModel
import plotly.graph_objects as go


def build_counterfactuals_text(original_row, cf_df):
    if cf_df is None or cf_df.empty:
        return ""

    lines = []
    for i, cf_row in cf_df.iterrows():
        changed = []
        for col in cf_row.index:
            if col in original_row.index and str(cf_row[col]) != str(original_row[col]):
                changed.append(f"{col}: {original_row[col]} -> {cf_row[col]}")
        if changed:
            lines.append(f"CF {i+1}: " + "; ".join(changed))
        else:
            lines.append(f"CF {i+1}: No changes detected.")

    return "\n".join(lines)


def build_student_summary(original_row, target_col):
    values = [f"{col}={original_row[col]}" for col in original_row.index if col != target_col]
    return f"Student profile: {'; '.join(values)}."


def clean_rag_section_title(title):
    """
    Clean Markdown markers from accordion titles while keeping the text readable.
    Examples:
    - "## 1. Interpretation" -> "1. Interpretation"
    - "**1. Risk factors:**" -> "1. Risk factors"
    """
    title = (title or "").strip()

    # Remove wrapping Markdown emphasis.
    title = re.sub(r"^\*+\s*", "", title)
    title = re.sub(r"\s*\*+$", "", title)

    # Remove extra heading markers, bullets and repeated spaces.
    title = re.sub(r"^#{1,6}\s*", "", title)
    title = re.sub(r"^\s*[-•]\s*", "", title)
    title = re.sub(r"\s+", " ", title).strip()

    # If the title ends with a Markdown bold marker or colon, make it cleaner.
    title = title.strip("*").strip()
    title = title[:-1].strip() if title.endswith(":") else title

    return title or "Recommendation section"


def is_rag_section_heading(line):
    """
    Detect section headings in RAG answers.

    It supports:
    - Markdown headings: ## 1. Interpretation
    - Numbered headings: 1. Interpretation
    - Bold numbered headings: **1. Interpretation**
    - Parenthesized headings: 1) Interpretation

    It intentionally ignores indented lines so normal nested lists are less likely
    to become accordion sections.
    """
    if not line or not line.strip():
        return None

    raw = line.rstrip()
    stripped = raw.strip()

    # Avoid converting indented nested lists into main accordion sections.
    if raw.startswith((" ", "\t")):
        return None

    # Markdown headings.
    heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
    if heading_match:
        return clean_rag_section_title(heading_match.group(1))

    # Remove wrapping bold/italic before testing numbered headings.
    simplified = stripped.strip()
    simplified = re.sub(r"^\*+\s*", "", simplified)
    simplified = re.sub(r"\s*\*+$", "", simplified)

    # Numbered top-level headings, e.g. "1. Interpretation" or "2) Risk factors".
    numbered_match = re.match(r"^(\d{1,2})[\.\)]\s+(.+?)\s*$", simplified)
    if numbered_match:
        candidate = f"{numbered_match.group(1)}. {numbered_match.group(2)}"
        return clean_rag_section_title(candidate)

    return None


def split_rag_markdown_sections(markdown_text):
    """
    Split an LLM/RAG Markdown answer into visual accordion sections.

    The LLM sometimes returns headings as Markdown:
        ## 1. Interpretation

    But it can also return plain numbered sections:
        1. Interpretation of the group
        2. Main risk factors

    This function detects both formats and turns them into collapsed accordion items.
    If no section heading is detected, the whole answer is shown in one collapsed item.
    """
    text = (markdown_text or "").strip()

    if not text:
        return [("Recommendation", "No recommendation generated.")]

    lines = text.splitlines()
    sections = []
    current_title = None
    current_lines = []

    def flush_section():
        nonlocal current_title, current_lines

        body = "\n".join(current_lines).strip()
        if current_title is not None:
            sections.append((current_title, body or "No additional details."))
        elif body:
            sections.append(("Overview", body))

        current_title = None
        current_lines = []

    for line in lines:
        section_title = is_rag_section_heading(line)

        if section_title:
            flush_section()
            current_title = section_title
        else:
            current_lines.append(line)

    flush_section()

    if not sections:
        sections = [("Recommendation", text)]

    return sections


def build_rag_card(recommendation_text, model_key=None, title="Individual RAG recommendation"):
    """
    Build a nicer RAG recommendation card.
    Each Markdown heading returned by the LLM is shown as a collapsible accordion item.
    """
    model_label = LLM_MODELS.get(model_key, {}).get("label", model_key or "LLM")
    sections = split_rag_markdown_sections(recommendation_text)

    accordion_items = []

    for i, (section_title, section_body) in enumerate(sections):
        # Use a Dash component as the accordion title so the section heading
        # keeps a strong visual weight even after Markdown markers are cleaned.
        accordion_title = html.Div(
            [
                html.Span(
                    section_title,
                    style={
                        "fontWeight": "850",
                        "color": "#0f172a",
                        "fontSize": "1rem",
                    },
                ),
                html.Span(
                    "Click to expand",
                    style={
                        "fontSize": "0.76rem",
                        "fontWeight": "700",
                        "color": "#64748b",
                        "background": "#f1f5f9",
                        "padding": "4px 9px",
                        "borderRadius": "999px",
                        "marginLeft": "10px",
                        "whiteSpace": "nowrap",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "gap": "12px",
                "width": "100%",
            },
        )

        accordion_items.append(
            dbc.AccordionItem(
                dcc.Markdown(
                    section_body,
                    className="rag-markdown",
                    link_target="_blank",
                    style={
                        "fontSize": "0.98rem",
                        "lineHeight": "1.7",
                        "color": "#1f2937",
                        "margin": "0",
                    },
                ),
                title=accordion_title,
                item_id=f"rag-section-{i}",
                style={
                    "border": "1px solid rgba(15, 23, 42, 0.08)",
                    "borderRadius": "12px",
                    "overflow": "hidden",
                    "marginBottom": "10px",
                },
            )
        )

    active_items = []  # Start with every section collapsed so the card does not take too much space.

    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "AI generated prescriptive recommendation",
                                    style={
                                        "fontSize": "0.76rem",
                                        "fontWeight": "800",
                                        "letterSpacing": "0.08em",
                                        "textTransform": "uppercase",
                                        "color": "#4f46e5",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.H4(
                                    title,
                                    style={
                                        "margin": "0",
                                        "fontSize": "1.25rem",
                                        "fontWeight": "800",
                                        "color": "#0f172a",
                                    },
                                ),
                                html.Div(
                                    "Evidence retrieved from the RAG system and adapted to the generated counterfactuals.",
                                    style={
                                        "fontSize": "0.9rem",
                                        "color": "#64748b",
                                        "marginTop": "4px",
                                    },
                                ),
                            ]
                        ),
                        html.Span(
                            model_label,
                            style={
                                "display": "inline-flex",
                                "alignItems": "center",
                                "padding": "7px 12px",
                                "borderRadius": "999px",
                                "background": "#0f172a",
                                "color": "#ffffff",
                                "fontSize": "0.82rem",
                                "fontWeight": "700",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "gap": "16px",
                    },
                ),
                style={
                    "background": "linear-gradient(135deg, #eef5ff 0%, #f8fbff 45%, #f6f3ff 100%)",
                    "borderBottom": "1px solid rgba(15, 23, 42, 0.08)",
                    "padding": "18px 22px",
                },
            ),
            dbc.CardBody(
                [
                    dbc.Accordion(
                        accordion_items,
                        active_item=active_items,
                        always_open=True,
                    )
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "padding": "18px",
                },
            ),
        ],
        style={
            "border": "1px solid rgba(15, 23, 42, 0.08)",
            "borderRadius": "18px",
            "overflow": "hidden",
            "background": "#ffffff",
            "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
        },
        class_name="mt-3",
    )


def build_rag_error_card(error_message, title="Individual RAG recommendation"):
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "AI generated prescriptive recommendation",
                                    style={
                                        "fontSize": "0.76rem",
                                        "fontWeight": "800",
                                        "letterSpacing": "0.08em",
                                        "textTransform": "uppercase",
                                        "color": "#dc2626",
                                        "marginBottom": "4px",
                                    },
                                ),
                                html.H4(
                                    title,
                                    style={
                                        "margin": "0",
                                        "fontSize": "1.25rem",
                                        "fontWeight": "800",
                                        "color": "#0f172a",
                                    },
                                ),
                            ]
                        ),
                        html.Span(
                            "Error",
                            style={
                                "display": "inline-flex",
                                "alignItems": "center",
                                "padding": "7px 12px",
                                "borderRadius": "999px",
                                "background": "#dc2626",
                                "color": "#ffffff",
                                "fontSize": "0.82rem",
                                "fontWeight": "700",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "gap": "16px",
                    },
                ),
                style={
                    "background": "#fff1f2",
                    "borderBottom": "1px solid rgba(220, 38, 38, 0.18)",
                    "padding": "18px 22px",
                },
            ),
            dbc.CardBody(
                dbc.Alert(
                    [
                        html.Strong("The recommendation could not be generated. "),
                        html.Span(str(error_message)),
                    ],
                    color="danger",
                    className="mb-0",
                ),
                style={"backgroundColor": "#ffffff"},
            ),
        ],
        style={
            "border": "1px solid rgba(220, 38, 38, 0.18)",
            "borderRadius": "18px",
            "overflow": "hidden",
            "background": "#ffffff",
            "boxShadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
        },
        class_name="mt-3",
    )


class AutoMLReportComponent(ExplainerComponent):
    """
    A component to display the AutoML report in the dashboard. Inherits from ExplainerComponent.
    """

    def __init__(self, explainer, title="AutoML Report", name=None,
                 subtitle="Compare all models and see which one fits best",
                 ML_report=None, **kwargs):
        super().__init__(explainer, title, name, **kwargs)
        self.ML_report = ML_report
        self.subtitle = subtitle

        # CSS para mejorar tablas del report sin tocar tamaños
        self._embedded_css = """
        <style>
          body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
          table { width:100% !important; border-collapse:collapse; border-spacing:0; border-radius:10px; overflow:hidden; }
          th,td { border:1px solid #eceff4; padding:8px 10px; text-align:center; }
          th { background:#eef2ff; color:#1f2d5a; font-weight:700; }
          tr:nth-child(even) td { background:#fafbff; }
          tr:hover td { background:#f2f6ff; }
          td:nth-child(5), th:nth-child(5) { font-weight:700; } /* resalta métrica si coincide */
        </style>
        """

    def layout(self):
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H3(self.title, className="m-0"),
                    html.Small(self.subtitle, className="text-muted")
                ])
            ]),
            dbc.CardBody([
                html.Iframe(
                    srcDoc=self._embedded_css + (self.ML_report.data or ""),
                    style={"width": "100%", "height": "1000px", "border": "0", "borderRadius": "10px"}
                )
            ])
        ], className="mb-3")

    
class SelectStudentComponent(ExplainerComponent):
    """
    Component for selecting a student for a classification model.
    """

    def __init__(self, explainer, title="Select Random Student", name=None,
                 subtitle="Select from list or pick at random",
                 index_dropdown=True,
                 pos_label=None, index=None, slider=None, labels=None,
                 pred_or_perc='predictions', description=None,
                 **kwargs):
        """
        Initialize SelectStudentComponent.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component
            - name (str): The name of the component.
            - subtitle (str): The subtitle of the component.
            - index_dropdown (bool): If True, a dropdown will be used for selecting the student index. Otherwise, an input field will be displayed.
            - pos_label (any): Positive class label to select.
            - index (str): Default index value.
            - slider (list of float or None): Slider range for selecting predicted probabilities
            - labels (list of str or None): Class labels to select
            - pred_or_perc (str): Whether to display predictions or percentiles
            - description (str or None): Description of the component
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Set name for index dropdown
        self.index_name = 'random-index-clas-index-'+self.name

        # Set slider range for selecting predicted probabilities
        if self.slider is None:
            self.slider = [0.0, 1.0]

        # Set class labels to select
        if self.labels is None:
            self.labels = self.explainer.labels

        # Hide labels if y is missing
        if self.explainer.y_missing:
            self.hide_labels = True

        # Initialize PosLabelSelector and IndexSelector
        self.selector = PosLabelSelector(
            explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(explainer, 'random-index-clas-index-'+self.name,
                                            index=index, index_dropdown=True, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap card
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                        # Add the title and subtitle
                        html.H3(f"Select student",
                                id='random-index-clas-title-'+self.name),
                        html.H6(self.subtitle, className='card-subtitle'),
                        ]),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        self.index_selector.layout()
                    ], width=8, md=8),
                    make_hideable(
                        dbc.Col([
                            self.selector.layout()
                        ], md=2),
                        hide=True),
                ], class_name="mb-2"),

                dbc.Row([
                    dbc.Col([
                        dbc.Button(f"Random student", color="primary",
                                   id='random-index-clas-button-'+self.name),
                    ], width=4, md=4),
                ], class_name="mb-2"),

                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            dbc.Label(f"Observed {self.explainer.target}:",
                                      id='random-index-clas-labels-label-'+self.name),
                            dcc.Dropdown(
                                id='random-index-clas-labels-'+self.name,
                                options=[{'label': lab, 'value': lab}
                                         for lab in self.explainer.labels],
                                multi=True,
                                value=self.labels),
                        ], width=8, md=8),
                        hide=True),
                    make_hideable(
                        dbc.Col([
                            dbc.Label(
                                "Range:", html_for='random-index-clas-pred-or-perc-'+self.name),
                            dbc.Select(
                                id='random-index-clas-pred-or-perc-'+self.name,
                                options=[
                                    {'label': 'probability',
                                        'value': 'predictions'},
                                    {'label': 'percentile',
                                        'value': 'percentiles'},
                                ],
                                value=self.pred_or_perc),
                        ], width=4,
                            id='random-index-clas-pred-or-perc-div-'+self.name),
                        hide=True)
                ], class_name="mb-2"),
                dbc.Row([
                    make_hideable(
                        dbc.Col([
                            html.Div([
                                dbc.Label(id='random-index-clas-slider-label-'+self.name,
                                          children="Predicted probability range:",
                                          html_for='prediction-range-slider-'+self.name),
                                dcc.RangeSlider(
                                    id='random-index-clas-slider-'+self.name,
                                    min=0.0, max=1.0, step=0.01,
                                    value=self.slider,  allowCross=False,
                                    marks={0.0: '0.0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6',
                                           0.8: '0.8', 1.0: '1.0'},
                                    tooltip={'always_visible': False})
                            ])
                        ]),
                        hide=True),
                ], justify="start"),
            ]),
        ], class_name="h-100")

    def component_callbacks(self, app):
        @app.callback(
            Output('random-index-clas-index-'+self.name, 'value'),
            [Input('random-index-clas-button-'+self.name, 'n_clicks')],
            [State('random-index-clas-slider-'+self.name, 'value'),
             State('random-index-clas-labels-'+self.name, 'value'),
             State('pos-label-'+self.name, 'value')])
        def update_index(n_clicks, slider_range, labels, pos_label):
            if n_clicks is None and self.index is not None:
                raise PreventUpdate

            return self.explainer.random_index(y_values=labels,
                                               pred_proba_min=slider_range[0], pred_proba_max=slider_range[1],
                                               return_str=True, pos_label=pos_label)


class SklearnCompatibleModel:
    def __init__(self, automl_model):
        # Usa el mejor modelo entrenado internamente por mljar
        self.model = automl_model.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CounterfactualsComponent(ExplainerComponent):
    """
    A component for generating counterfactual scenarios for a given student in a dataset. Inherits from ExplainerComponent.
    """

    def __init__(self, explainer, title="Counterfactuals scenarios", name=None,
                 subtitle="What can a student improve?",
                 index_dropdown=True, index=None, dataframe=None, trained_model=None, **kwargs):
        """
        Initializes the CounterfactualsComponent instance.

        Args:
            - explainer (ClassifierExplainer): Explainer instance containing dataset and trained model.
            - title (str): Title of the component.
            - name (str): Name of the component (for internal use).
            - subtitle (str): Subtitle of the component.
            - index_dropdown (bool): If True, a dropdown will be used for selecting the student index. Otherwise, an input field will be displayed.
            - index (str): Default index value.
            - dataframe (Dataframe): Pandas dataframe containing the dataset.
            - trained_model (any): Trained model used for generating counterfactuals.
            - **kwargs: Optional keyword arguments.
        """

        # Call the parent constructor
        super().__init__(explainer, title, name)

        # Set the dataframe attribute
        self.dataframe = dataframe

        # Set the trained_model attribute
        self.trained_model = trained_model

        # Initialize IndexSelector
        self.index_selector = IndexSelector(explainer, 'random-index-clas-index-'+self.name,
                                            index=index, index_dropdown=index_dropdown, **kwargs)

    def layout(self):
        """
        Layout of the component.

        Returns:
            - The layout of the component wrapped in a Bootstrap card.
        """

        # Create a Bootstrap card
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                        # Add the title and subtitle
                        html.H3(f"Select student and number of counterfactual scenarios",
                                id='random-index-clas-title-'+self.name),
                        html.H6(self.subtitle, className='card-subtitle'),
                        ]),
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        # Add the index selector
                        self.index_selector.layout()
                    ], width=8, md=8),
                ], class_name="mb-2"),

                dbc.Row([
                    dbc.Col([
                        # Add the input field for the number of scenarios to generate
                        dbc.Input(id="input",
                                  placeholder="Enter the number of scenarios between 1 and 10",
                                  type="number",
                                  debounce=True,
                                  min=1,
                                  max=10,
                                  value=None)
                    ], width=8, md=8),
                ], class_name="mb-2"),

                dbc.Row([
                    dbc.Col([
                        # Add the button to generate the scenarios
                        dbc.Button(f"Generate scenarios",
                                   color="primary", id="button"),
                    ], width=4, md=4),
                ], class_name="mb-2"),

                html.Br(),
                html.Br(),

                dbc.Spinner(
                    dbc.Row([
                        # Add a data table for displaying the original data
                        html.H3(f"Original", id='original-title', hidden=True),
                        dash_table.DataTable(
                            id='original-tbl', data=None, style_table={'overflowX': 'scroll'})
                    ], class_name="mb-2")),

                html.Br(),

                dbc.Spinner(
                    dbc.Row([
                        # Add a data table for displaying the generated counterfactuals
                        html.H3(f"Counterfactuals",
                                id='counterfactuals-title', hidden=True),
                        dash_table.DataTable(
                            id='counterfactuals-tbl', data=None, style_table={'overflowX': 'scroll'})
                    ], class_name="mb-2")),

                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H5("Visualization of the generated counterfactuals"),
                        html.Div(id='cf-graphs')
                    ]),
                ]),

                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("LLM for RAG recommendation"),
                        dcc.Dropdown(
                            id='rag-llm-dropdown-'+self.name,
                            options=[{'label': cfg['label'], 'value': key} for key, cfg in LLM_MODELS.items()],
                            value=list(LLM_MODELS.keys())[0],
                            clearable=False
                        ),
                    ], width=6, md=6)
                ], class_name="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            id='rag-loading-'+self.name,
                            children=html.Div(
                                id='rag-output-'+self.name,
                                style={'display': 'none'}
                            ),
                            type='default'
                        )
                    ])
                ]),
            ]),
        ], class_name="h-100")

    def component_callbacks(self, app):
        @app.callback(
            Output('counterfactuals-tbl', 'data'),
            Output('counterfactuals-tbl', 'columns'),
            Output('original-tbl', 'data'),
            Output('original-tbl', 'columns'),
            Output('counterfactuals-title', 'hidden'),
            Output('original-title', 'hidden'),
            Output('cf-graphs', 'children'),
            Output('rag-output-'+self.name, 'children'),
            Output('rag-output-'+self.name, 'style'),
            [State('random-index-clas-index-'+self.name, 'value'),
             State('input', 'value'),
             State('rag-llm-dropdown-'+self.name, 'value')],
            [Input('button', 'n_clicks')]
        )
        def generate_counterfactuals(dataIndex, input, rag_model_key, n_clicks):
            """
            Generate counterfactual scenarios given a row index and a desired number of counterfactuals.
            This version prepares DiCE inputs using one single, normalized dataframe so the query instance
            cannot fall outside the ranges/categories known by DiCE.
            """

            print("📥 Activated Callback:")
            print("📌 Received DataIndex:", dataIndex)
            print("✅ Clicked button:", n_clicks)

            empty_response = ([], [], [], [], True, True, [], "", {'display': 'none'})

            if n_clicks is None or dataIndex is None or input is None:
                print("⚠️ Missing input data")
                return empty_response

            try:
                total_cfs = int(input)
            except Exception:
                return [], [], [], [], True, True, [
                    dbc.Alert("Please enter a valid number of counterfactual scenarios.", color="warning")
                ], "", {'display': 'none'}

            total_cfs = max(1, min(total_cfs, 10))

            print("📌 DataIndex type :", type(dataIndex))
            print("📊 First 5 índices of the dataframe:", self.dataframe.index[:5])

            # -----------------------------
            # 1) Build one clean dataframe for DiCE
            # -----------------------------
            df = self.dataframe.copy()
            df.index = df.index.map(str)
            key = str(dataIndex)

            print("✅ ¿Is index in the dataframe?", key in df.index)

            if key not in df.index:
                return [], [], [], [], True, True, [
                    dbc.Alert(f"Student id '{key}' not found in current data.", color="warning")
                ], "", {'display': 'none'}

            target_col = df.columns[-1]
            feature_cols = [c for c in df.columns if c != target_col]

            # Make sure target is numeric for DiCE/classification.
            df[target_col] = df[target_col].replace({
                "Dropout": 0,
                "No dropout": 1,
                "dropout": 0,
                "no dropout": 1
            })

            # Drop rows with missing target.
            df = df.dropna(subset=[target_col]).copy()

            # Normalize target to int when possible.
            try:
                df[target_col] = df[target_col].astype(int)
            except Exception:
                pass

            # Query instance and original row.
            student_data = df.loc[[key]].copy()
            query_instance = student_data[feature_cols].copy()

            # -----------------------------
            # 2) Align dtypes between dataframe and query
            # -----------------------------
            # DiCE validates the query against the dataframe ranges/categories.
            # We keep the same dtypes and append the query row back to the DiCE dataframe
            # so categorical and numeric checks always include the selected student.
            dice_dataframe = df.copy()

            for col in feature_cols:
                # If the column is numeric in the dataframe, make query numeric too.
                if pd.api.types.is_numeric_dtype(dice_dataframe[col]):
                    dice_dataframe[col] = pd.to_numeric(dice_dataframe[col], errors="coerce")
                    query_instance[col] = pd.to_numeric(query_instance[col], errors="coerce")
                else:
                    # Keep categorical values comparable.
                    dice_dataframe[col] = dice_dataframe[col].astype(str)
                    query_instance[col] = query_instance[col].astype(str)

            # Remove rows that cannot be used by DiCE, but never lose the selected query row.
            dice_dataframe = dice_dataframe.dropna(subset=feature_cols + [target_col]).copy()

            query_with_target = query_instance.copy()
            query_with_target[target_col] = student_data[target_col].iloc[0]

            # Append the query explicitly. This avoids:
            # ValueError: ('Feature', feature, 'has a value outside the dataset.')
            dice_dataframe = pd.concat([dice_dataframe, query_with_target], ignore_index=True)

            # -----------------------------
            # 3) Select feature types and actionable features
            # -----------------------------
            continuous_features = [
                col for col in feature_cols
                if pd.api.types.is_float_dtype(dice_dataframe[col])
            ]

            categorical_features = [
                col for col in feature_cols
                if col not in continuous_features
            ]

            # Avoid non-prescriptive or immutable variables in counterfactual recommendations.
            non_actionable_keywords = [
                "marital", "nationality", "gender", "age at enrollment",
                "application mode", "application order", "course",
                "daytime/evening", "previous qualification",
                "mother", "father", "educational special needs",
                "international", "displaced"
            ]

            preferred_actionable_keywords = [
                "approved", "grade", "enrolled", "evaluations",
                "credited", "tuition fees", "scholarship", "curricular units"
            ]

            def is_non_actionable(col):
                col_l = str(col).lower()
                return any(k in col_l for k in non_actionable_keywords)

            actionable_features = [
                col for col in feature_cols
                if not is_non_actionable(col)
                and any(k in str(col).lower() for k in preferred_actionable_keywords)
            ]

            # Fallback if names are different in another dataset.
            if not actionable_features:
                actionable_features = [
                    col for col in feature_cols
                    if not is_non_actionable(col)
                ]

            # DiCE needs at least one feature to vary.
            if not actionable_features:
                actionable_features = feature_cols

            # Explicit permitted ranges for continuous features.
            # We slightly expand the min/max to avoid strict boundary errors.
            permitted_range = {}
            for col in continuous_features:
                col_values = pd.to_numeric(dice_dataframe[col], errors="coerce").dropna()
                if col_values.empty:
                    continue

                q_val = pd.to_numeric(query_instance[col], errors="coerce").iloc[0]
                min_v = min(float(col_values.min()), float(q_val))
                max_v = max(float(col_values.max()), float(q_val))

                if min_v == max_v:
                    eps = 1e-6
                else:
                    eps = max((max_v - min_v) * 0.001, 1e-6)

                permitted_range[col] = [min_v - eps, max_v + eps]

            print("🧩 Continuous features:", continuous_features)
            print("🧩 Categorical features:", categorical_features)
            print("🎯 Features to vary:", actionable_features)
            print("📌 Query instance:")
            print(query_instance)

            # Create a data object using the cleaned dataframe.
            data = dml.Data(
                dataframe=dice_dataframe,
                continuous_features=continuous_features,
                categorical_features=categorical_features,
                outcome_name=target_col
            )

            # Simulación manual de la clase ModelTypes
            class ModelTypes:
                Classifier = "classifier"
                Regressor = "regressor"

            from dice_ml.model_interfaces.base_model import BaseModel

            class CustomSklearnModel(BaseModel):
                def __init__(self, model):
                    super().__init__(model=model, backend="sklearn")
                    self.model_type = ModelTypes.Classifier

                def predict(self, input_data):
                    input_data = input_data[feature_cols]
                    return self.model.predict(input_data)

                def predict_proba(self, input_data):
                    input_data = input_data[feature_cols]
                    return self.model.predict_proba(input_data)

            class SklearnCompatibleModel:
                def __init__(self, clf):
                    self.model = clf
                    self.classes_ = [0, 1]

                def predict(self, X):
                    X = X[feature_cols]
                    return self.model.predict(X)

                def predict_proba(self, X):
                    X = X[feature_cols]
                    return self.model.predict_proba(X)

            # Detect if the model has submodel.
            if hasattr(self.trained_model, "model"):
                print("🔁 Received model is AutoML container, using .model")
                model_to_use = self.trained_model.model
            else:
                print("✅ The received model is final (RandomForest, etc.), using it directly.")
                model_to_use = self.trained_model

            wrapped_model = SklearnCompatibleModel(model_to_use)
            model = CustomSklearnModel(model=wrapped_model)

            print("🚀 wrapped_model.classes_:", wrapped_model.classes_)
            print("🚀 wrapped_model.predict_proba:", wrapped_model.predict_proba(query_instance))

            # Create a Dice object using the data and model objects.
            exp = dml.Dice(data, model, method="random")

            # -----------------------------
            # 4) Generate counterfactuals
            # -----------------------------
            # Goal of the individual counterfactuals:
            # minimize the student's dropout probability.
            # Project encoding: 0 = Dropout, 1 = No dropout.
            desired_no_dropout_class = 1
            candidate_total_cfs = max(total_cfs, 10)

            try:
                print("🎯 Original P(Dropout):", model.predict_proba(query_instance)[0][0])
                print("🎯 Generating candidate CFs towards No dropout:", candidate_total_cfs)

                try:
                    dice_exp = exp.generate_counterfactuals(
                        query_instance,
                        total_CFs=candidate_total_cfs,
                        desired_class=desired_no_dropout_class,
                        features_to_vary=actionable_features,
                        permitted_range=permitted_range if permitted_range else None
                    )
                except Exception as first_error:
                    # If DiCE cannot generate a larger candidate pool, retry with the requested amount.
                    if candidate_total_cfs != total_cfs:
                        print(f"⚠️ Could not generate {candidate_total_cfs} CF candidates. Retrying with {total_cfs}: {first_error}")
                        dice_exp = exp.generate_counterfactuals(
                            query_instance,
                            total_CFs=total_cfs,
                            desired_class=desired_no_dropout_class,
                            features_to_vary=actionable_features,
                            permitted_range=permitted_range if permitted_range else None
                        )
                    else:
                        raise
            except ValueError as e:
                print(f"❌ Error generating counterfactuals: {e}")
                original_display = student_data.copy()
                original_display[target_col] = original_display[target_col].replace({0: "Dropout", 1: "No dropout"})

                original_data = original_display.to_dict("records")
                original_columns = [{"name": i, "id": i} for i in original_display.columns]

                return [], [], original_data, original_columns, True, False, [
                    dbc.Alert(
                        [
                            html.Strong("DiCE could not generate counterfactuals for this student. "),
                            html.Span(str(e)),
                            html.Br(),
                            html.Small(
                                "The input row was normalized and checked, but DiCE still rejected one feature. "
                                "Try another student or reduce immutable/categorical variables."
                            )
                        ],
                        color="warning"
                    )
                ], "", {'display': 'none'}
            except Exception as e:
                print(f"❌ Unexpected error generating counterfactuals: {e}")
                original_display = student_data.copy()
                original_display[target_col] = original_display[target_col].replace({0: "Dropout", 1: "No dropout"})

                original_data = original_display.to_dict("records")
                original_columns = [{"name": i, "id": i} for i in original_display.columns]

                return [], [], original_data, original_columns, True, False, [
                    dbc.Alert(f"Unexpected error generating counterfactuals: {e}", color="danger")
                ], "", {'display': 'none'}

            if not dice_exp.cf_examples_list or dice_exp.cf_examples_list[0].final_cfs_df is None or dice_exp.cf_examples_list[0].final_cfs_df.empty:
                print("❌ No counterfactuals generated.")

                original_display = student_data.copy()
                original_display[target_col] = original_display[target_col].replace({0: "Dropout", 1: "No dropout"})

                original_data = original_display.to_dict("records")
                original_columns = [{"name": i, "id": i} for i in original_display.columns]

                return [], [], original_data, original_columns, True, False, [
                    dbc.Alert(
                        "DiCE did not find valid counterfactuals for this student with the current constraints.",
                        color="warning"
                    )
                ], "", {'display': 'none'}

            cf_object_to_csv = dice_exp.cf_examples_list[0].final_cfs_df.copy()

            # Some DiCE versions return the target as a column, others do not.
            if target_col not in cf_object_to_csv.columns:
                predicted_cf = model.predict(cf_object_to_csv[feature_cols])
                cf_object_to_csv[target_col] = predicted_cf

            # Rank counterfactuals by the lowest predicted dropout probability.
            # This keeps the most prescriptive scenarios first.
            try:
                cf_features = cf_object_to_csv[feature_cols].copy()
                cf_dropout_probas = model.predict_proba(cf_features)[:, 0]
                cf_object_to_csv["Predicted P(Dropout)"] = cf_dropout_probas
                cf_object_to_csv = (
                    cf_object_to_csv
                    .sort_values("Predicted P(Dropout)", ascending=True)
                    .head(total_cfs)
                    .reset_index(drop=True)
                )
                print("✅ Individual CFs sorted by lowest P(Dropout):")
                print(cf_object_to_csv[["Predicted P(Dropout)"]])
            except Exception as e:
                print(f"⚠️ Could not rank individual CFs by P(Dropout): {e}")

            # Keep only known columns, target and optional risk score, avoiding index/extra columns.
            final_cols = [c for c in feature_cols if c in cf_object_to_csv.columns] + [target_col]
            if "Predicted P(Dropout)" in cf_object_to_csv.columns:
                final_cols.append("Predicted P(Dropout)")
            cf_object_to_csv = cf_object_to_csv[final_cols]

            # Replace binary values with labels.
            student_display = student_data.copy()
            student_display[target_col] = student_display[target_col].replace({0: "Dropout", 1: "No dropout"})
            cf_display = cf_object_to_csv.copy()
            cf_display[target_col] = cf_display[target_col].replace({0: "Dropout", 1: "No dropout"})

            original_data = student_display.to_dict("records")
            original_columns = [{"name": i, "id": i} for i in student_display.columns]
            counterfactuals_data = cf_display.to_dict("records")
            counterfactuals_columns = [{"name": i, "id": i} for i in cf_display.columns]

            print("✅ Counterfactuals generated correctly:")
            print(cf_object_to_csv)

            # -----------------------------
            # 5) Graphs and tables
            # -----------------------------
            original_proba = model.predict_proba(query_instance)[0][0]  # P(Dropout)

            cf_probas = [
                model.predict_proba(pd.DataFrame([row], columns=feature_cols))[0][0]
                for _, row in cf_object_to_csv[feature_cols].iterrows()
            ]

            labels = ["Original"] + [f"CF {i+1}" for i in range(len(cf_probas))]
            values = [original_proba] + cf_probas

            fig = go.Figure(go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.2%}" for v in values],
                textposition="outside"
            ))

            fig.update_layout(
                title="Dropout probability (original vs counterfactuals)",
                yaxis_title="P(Dropout)",
                height=400
            )

            graph = dcc.Graph(figure=fig)

            numeric_features = query_instance.select_dtypes(include=["number"]).columns

            features = [
                col for col in numeric_features
                if col in cf_object_to_csv.columns
                and any(cf[col] != query_instance[col].iloc[0] for _, cf in cf_object_to_csv.iterrows())
            ]

            if not features:
                radar_graph = html.Div([
                    html.Br(),
                    html.Div(
                        "⚠️ There are no modified or valid numeric variables for the radar chart.",
                        style={"color": "orange", "fontWeight": "bold"}
                    )
                ])
                change_table = html.Div("No numeric feature changes to show.")
                impact_graph = html.Div("No modified variables, the impact cannot be estimated.")
            else:
                radar_fig = go.Figure()

                for i, row in cf_object_to_csv[features].iterrows():
                    radar_fig.add_trace(go.Scatterpolar(
                        r=row.values,
                        theta=features,
                        fill='toself',
                        name=f"CF {i+1}"
                    ))

                radar_fig.add_trace(go.Scatterpolar(
                    r=query_instance[features].iloc[0],
                    theta=features,
                    fill='toself',
                    name='Original',
                    line=dict(color='black', dash='dot')
                ))

                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title="Radar chart: Original vs Counterfactuals (Numeric features modified)",
                    showlegend=True,
                    height=500
                )

                radar_graph = dcc.Graph(figure=radar_fig)

                original_row = query_instance.iloc[0]
                cf_rows = cf_object_to_csv[feature_cols]

                comparison_df = pd.DataFrame({'Original': original_row})

                for i in range(len(cf_rows)):
                    comparison_df[f'CF {i+1}'] = cf_rows.iloc[i]

                changed_only = comparison_df.loc[
                    (comparison_df != comparison_df['Original'].values.reshape(-1, 1)).any(axis=1)
                ]

                changed_with_var = changed_only.reset_index().rename(columns={'index': 'Variable'})

                change_table = dash_table.DataTable(
                    columns=[{'name': c, 'id': c} for c in changed_with_var.columns],
                    data=changed_with_var.to_dict('records'),
                    style_data_conditional=[
                        {
                            'if': {'column_id': col, 'filter_query': f'{{{col}}} != {{Original}}'},
                            'backgroundColor': '#d2f8d2'
                        }
                        for col in changed_with_var.columns
                        if col not in ['Original', 'Variable']
                    ] + [
                        {'if': {'column_id': 'Original'}, 'backgroundColor': '#f0f0f0', 'fontWeight': 'bold'},
                        {'if': {'column_id': 'Variable'}, 'textAlign': 'left', 'fontWeight': '600'}
                    ],
                    style_table={'overflowX': 'scroll'},
                    style_cell={'textAlign': 'center'},
                )

                variable_impact = {}

                for _, cf_row in cf_object_to_csv.iterrows():
                    cf_row_clean = cf_row[feature_cols]

                    changed_vars = [
                        col for col in feature_cols
                        if cf_row_clean[col] != original_row[col]
                    ]

                    if not changed_vars:
                        continue

                    prob_orig = model.predict_proba(pd.DataFrame([original_row], columns=feature_cols))[0][0]
                    prob_cf = model.predict_proba(pd.DataFrame([cf_row_clean], columns=feature_cols))[0][0]
                    delta_prob = prob_cf - prob_orig

                    impact_per_var = delta_prob / len(changed_vars)

                    for var in changed_vars:
                        variable_impact[var] = variable_impact.get(var, 0) + impact_per_var

                if variable_impact:
                    impact_df = pd.DataFrame(
                        list(variable_impact.items()),
                        columns=["Variable", "Cumulative impact"]
                    )
                    impact_df.sort_values("Cumulative impact", ascending=True, inplace=True)

                    impact_fig = go.Figure(go.Bar(
                        x=impact_df["Cumulative impact"],
                        y=impact_df["Variable"],
                        orientation='h'
                    ))

                    impact_fig.update_layout(
                        title="Cumulative impact per modified feature (all the counterfactuals)",
                        xaxis_title="Total contribution to probability change",
                        yaxis_title="Variable",
                        height=450,
                        margin=dict(l=250),
                        yaxis=dict(tickfont=dict(size=11))
                    )

                    impact_graph = dcc.Graph(figure=impact_fig)
                else:
                    impact_graph = html.Div(
                        "⚠️ There are no modified variables, the impact cannot be estimated."
                    )

            recommendation_component = ""
            recommendation_style = {'display': 'none'}

            try:
                original_row = student_data.iloc[0]
                cf_display_df = cf_display.copy()
                counterfactuals_text = build_counterfactuals_text(original_row, cf_display_df)
                group_description = build_student_summary(original_row, target_col)

                recommendation_text = generate_rag_recommendations(
                    group_description=group_description,
                    counterfactuals_text=counterfactuals_text,
                    model_key=rag_model_key
                )

                recommendation_component = build_rag_card(
                    recommendation_text=recommendation_text,
                    model_key=rag_model_key,
                    title="Individual RAG recommendation"
                )
                recommendation_style = {'display': 'block'}

            except Exception as e:
                recommendation_component = build_rag_error_card(
                    error_message=e,
                    title="Individual RAG recommendation"
                )
                recommendation_style = {'display': 'block'}

            return (
                counterfactuals_data,
                counterfactuals_columns,
                original_data,
                original_columns,
                False,
                False,
                [
                    graph,
                    radar_graph,
                    html.Br(),
                    html.H5("Changes in features"),
                    change_table,
                    html.Br(),
                    html.H5("Predicted impact of each modified feature"),
                    impact_graph
                ],
                recommendation_component,
                recommendation_style
            )


class TimerComponent(ExplainerComponent):
    def __init__(self, explainer, title="Timer", name=None, **kwargs):
        super().__init__(explainer, title, name)

    def layout(self):
        return dbc.CardBody([
            dbc.Row([
                    dcc.Store(id="time"),
                    html.H2(id="display-time", children='Start timer'),
                    html.Br(),
                    dbc.Button(f"Start", color="primary", id="start-button",
                               style={'textAlign': 'center', 'width': '10%', }),
                    html.Br(),
                    dbc.Button(f"Stop", color="primary", id="stop-button",
                               style={'textAlign': 'center', 'width': '10%'}),
                    ], class_name="mb-2"),
        ], class_name="h-100")

    def component_callbacks(self, app):
        @app.callback(
            Output('display-time', 'children'),
            Output('time', 'data'),
            Output('stop-button', 'n_clicks'),
            Input('start-button', 'n_clicks'),
            Input('stop-button', 'n_clicks'),
            State('time', 'data'),
            prevent_initial_call=True
        )
        def update_timer(n_clicks_start, n_clicks_stop, start_time):
            if n_clicks_start and not n_clicks_stop:
                started_timer = time.time()
                return "The timer has started.", started_timer, None

            stopped_timer = time.time()
            try:
                elapsed_time = stopped_timer - start_time
            except:
                return 'Start timer', None, None

            minutes, seconds = divmod(elapsed_time, 60)
            hours, minutes = divmod(minutes, 60)

            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", None, None
