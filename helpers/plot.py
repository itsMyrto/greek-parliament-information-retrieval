import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def radar_chart(emotion_scores, emotions):
    """
    Display a radar chart representing emotions for each person.
    Parameters:
        emotion_scores (list of dicts): Emotion scores for each person.
        emotions (list): List of emotion categories.
    """
    
    fig = go.Figure()
    
    for personCount in emotion_scores:
        fig.add_trace(go.Scatterpolar(
            r=[personCount["emotions"][emotion] for emotion in emotions],
            theta=emotions,
            fill='toself',
            name=personCount["member_name"]
        ))
    
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
    

def bar_chart(positivity_scores):
    """
    Display a grouped bar chart showing positivity and negativity scores for each politician.
    Parameters:
    positivity_scores (dict): Dictionary containing positivity and negativity scores for each politician.
    """
    
    politicians = []
    positive_scores = []
    negative_scores = []
    for politician in positivity_scores:
        politicians.append(politician)
        positive_scores.append(positivity_scores[politician][0])
        negative_scores.append(positivity_scores[politician][1])

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


def displayPlots(counts: list):
    """ Display the plots """

    positivitiesDict = {
        item["member_name"]: (item["positivity-negativity"][0], item["positivity-negativity"][1]) for item in counts
    }
    
    subjectiviesDict = {
        item["member_name"]: (item["subjectivity-objectivity"][0], item["subjectivity-objectivity"][1]) for item in counts
    }
    
    radar_chart(counts, list(counts[0]["emotions"].keys()))
    bar_chart(positivitiesDict)
    

