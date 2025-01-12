"""
dashboard.py

This module creates an interactive dashboard to visualize mutual fund predictions, risks, and trends.
Author: Satej
"""

import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Load data for the dashboard
def load_dashboard_data(file_path):
    """
    Load processed data for the dashboard.

    Args:
        file_path (str): Path to the processed data file.

    Returns:
        pd.DataFrame: Processed data.
    """
    return pd.read_csv(file_path, index_col='date', parse_dates=True)


# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Mutual Fund Performance Dashboard", style={'textAlign': 'center'}),

    # Dropdown to select a metric
    html.Div([
        html.Label("Select Metric:"),
        dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'Net Asset Value (NAV)', 'value': 'nav'},
                {'label': 'Volatility', 'value': 'volatility'},
                {'label': 'Sharpe Ratio', 'value': 'sharpe_ratio'}
            ],
            value='nav'
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    # Graph to display selected metric
    dcc.Graph(id='metric-graph'),

    # Date range slider
    html.Div([
        dcc.RangeSlider(
            id='date-slider',
            min=0,
            max=100,  # Placeholder
            step=1,
            marks={},
            value=[0, 100]
        )
    ], style={'margin': '40px'}),
])


# Callback to update the graph based on dropdown and slider
@app.callback(
    Output('metric-graph', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('date-slider', 'value')]
)
def update_graph(selected_metric, date_range):
    """
    Update the graph based on the selected metric and date range.

    Args:
        selected_metric (str): Metric selected from dropdown.
        date_range (list): Start and end indices for date range.

    Returns:
        go.Figure: Updated plotly figure.
    """
    filtered_data = data.iloc[date_range[0]:date_range[1]]
    trace = go.Scatter(x=filtered_data.index, y=filtered_data[selected_metric], mode='lines', name=selected_metric)
    layout = go.Layout(title=f"{selected_metric} Over Time", xaxis={'title': 'Date'}, yaxis={'title': selected_metric})
    return {'data': [trace], 'layout': layout}


if __name__ == "__main__":
    # Load data for the dashboard
    input_file = "satej/engineered_mutual_fund_data.csv"
    data = load_dashboard_data(input_file)

    # Update date slider range and marks dynamically
    app.layout['children'][2]['children'][0]['min'] = 0
    app.layout['children'][2]['children'][0]['max'] = len(data) - 1
    app.layout['children'][2]['children'][0]['marks'] = {
        i: str(date.date()) for i, date in enumerate(data.index[::len(data)//10])
    }

    print("Running dashboard...")
    app.run_server(debug=True)
