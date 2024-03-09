
# go to: http://localhost:8050/

import dash
from dash import dcc
import dash_html_components as html
from Text_mining import total_reviews, negative_reviews, positive_reviews  # Importing variables from data.py

# Initialize the Dash app
app = dash.Dash(__name__)

# Define styles
big_title_style = {
    'textAlign': 'center',
    'marginBottom': '20px',
    'fontFamily': 'Arial, sans-serif',
    'fontSize': '32px',
    'color': '#333'  # Dark gray
}

italic_style = {
    'font-style': 'italic'  # Set the font style to italic
}

label_style = {
    'textAlign': 'center',
    'marginTop': '10px',
    'fontFamily': 'Arial, sans-serif',
    'fontSize': '18px',
    'color': '#666'  # Medium gray
}

image_style = {
    'width': '40%',
    'margin': '10px',
    'borderRadius': '10px'
}

# Define the layout
app.layout = html.Div(
    style={'backgroundColor': '#f7f7f7'},  # Light gray background
    children=[
        html.H1([html.Span("The Boy and the Heron", style=italic_style)," movie critics reviews text mining"], style=big_title_style),
        html.Div([
            html.H3(f"Total number of reviews: {total_reviews}", style=label_style),
            html.H3(f"Total number of negative reviews: {negative_reviews}", style=label_style),
            html.H3(f"Total number of positive reviews: {positive_reviews}", style=label_style),
        ]),
        html.Div([
            html.Img(src='assets/chart1.png', style=image_style),
            html.Img(src='assets/Word Cloud for negative reviews.png', style=image_style),
            html.Img(src='assets/Word Cloud for positive reviews.png', style=image_style),
            html.Img(src='assets/Word Cloud of bigrams for negative reviews.png', style=image_style),
            html.Img(src='assets/Word Cloud of bigrams for positive reviews.png', style=image_style),
            html.Img(src='assets/chart6.png', style=image_style),
            html.Img(src='assets/chart7.png', style=image_style),
            html.Img(src='assets/chart8.png', style=image_style),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})
    ]
)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

