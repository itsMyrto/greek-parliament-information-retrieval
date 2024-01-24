import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def radar_chart(emotion_scores):
    df = pd.DataFrame(dict(
        r=emotion_scores,
        theta=["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    ))

    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself', line=dict(color='darkorange'))

    # Customize layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                tickfont=dict(size=16),
                tickangle=45,
                tickmode='array',
                gridcolor='lightgrey',
            ),
        ),
        showlegend=True
    )

    fig.show()

def bar_chart(positive_scores, negative_scores):

    politicians = ["Varoufakis", "Mitsotakis", "Tsipras", "Koutsoumpas", "Kasidiaris", "Velopoulos"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=politicians,
        y=positive_scores,
        name='Overall Positivity',
        marker_color='#9ADE7B'
    ))
    fig.add_trace(go.Bar(
        x=politicians,
        y=negative_scores,
        name='Overall Negativity',
        marker_color='#F05941'
    ))

    fig.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        xaxis_tickfont_size=14,
        legend=dict(
            title='Sentiment',
            font=dict(size=16)
        ),
        plot_bgcolor='white',
        bargap=0.2,
        bargroupgap=0.1
    )

    fig.show()




