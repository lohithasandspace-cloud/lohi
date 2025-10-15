import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

# Sample dataset for dashboard metrics
data = pd.DataFrame({
    "Category": ["Real", "Fake"],
    "Count": [120, 80]
})

def create_dashboard(server):
    dash_app = dash.Dash(
        server=server,
        name="Dashboard",
        url_base_pathname='/dash/',  # THIS MAKES DASH ACCESSIBLE AT /dash/
        external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
    )

    dash_app.layout = html.Div(style={'backgroundColor':'#1e1e2f', 'color':'#fff', 'padding':'20px'}, children=[
        html.H1("Business Dashboard", style={'textAlign':'center', 'color':'#00ffff'}),

        # KPI cards
        html.Div(style={'display':'flex', 'justifyContent':'space-around'}, children=[
            html.Div(style={'backgroundColor':'#2e2e3e','padding':'20px','borderRadius':'10px','width':'200px','textAlign':'center'}, children=[
                html.H3("Total Articles"),
                html.H2(f"{data['Count'].sum()}")
            ]),
            html.Div(style={'backgroundColor':'#2e2e3e','padding':'20px','borderRadius':'10px','width':'200px','textAlign':'center'}, children=[
                html.H3("Real %"),
                html.H2(f"{data.loc[data['Category']=='Real','Count'].values[0]/data['Count'].sum()*100:.1f}%")
            ]),
            html.Div(style={'backgroundColor':'#2e2e3e','padding':'20px','borderRadius':'10px','width':'200px','textAlign':'center'}, children=[
                html.H3("Fake %"),
                html.H2(f"{data.loc[data['Category']=='Fake','Count'].values[0]/data['Count'].sum()*100:.1f}%")
            ]),
        ]),

        html.Hr(style={'borderColor':'#555'}),

        # Pie chart
        dcc.Graph(
            figure=px.pie(data, names='Category', values='Count', hole=0.4,
                          color_discrete_sequence=['#00ff00','#ff3333'])
        )
    ])
