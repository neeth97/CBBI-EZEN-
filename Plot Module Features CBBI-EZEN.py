#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[9]:


import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d


# # Sample data for Scatter Plots

# In[10]:


df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'z': np.random.randn(100),
    'color_label': np.random.choice(['Category 1', 'Category 2'], size=100),
    'shape_label': np.random.choice(['circle', 'square'], size=100),
    'opacity_label': np.random.rand(100),
    'label': [f'point_{i}' for i in range(100)],
    'error': np.random.rand(100) * 0.2  # Random error for error bars
})


# # Global configuration for the plot layout

# In[11]:


def set_global_plotly_config():
    """
    Set global configuration for Plotly plots, which can be overridden for individual plots.
    """
    layout = {
        'font': {
            'color': 'black',  # Text color
            'size': 14         # Font size
        },
        'plot_bgcolor': 'white',  # Background color
        'paper_bgcolor': 'white',
        'legend': {
            'x': 1,
            'y': 1,
            'xanchor': 'right',
            'yanchor': 'top',
            'borderwidth': 1
        },
        # Using BPMC color palette as the default
        'colorway': ['#006E96', '#00263D', '#B8D87A', '#8E847A', '#FFFFFF', '#EB6852', '#983E53'],
    }
    
    return layout


# # Scatter plot with 2D and advanced features

# In[12]:


# Scatter plot with 2D and advanced features

def set_global_plotly_config():
    """
    Set global plotly configuration for consistent styling.
    
    Returns:
        layout (dict): Global layout configuration for plotly figures.
    """
    layout = dict(
        title="2D Scatter Plot with Advanced Features",
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        plot_bgcolor='rgba(245, 245, 245, 1)',
        showlegend=False,
    )
    return layout

def create_2d_scatter(df):
    """
    Create a 2D scatter plot with advanced customization options, including color, shape, opacity, and linear regression.
    
    Args:
        df (pd.DataFrame): Input data frame in long or wide format.
    
    Returns:
        fig (plotly.graph_objs.Figure): The configured scatter plot.
    """
    # Create a mapping for shapes
    shape_map = {'circle': 'circle', 'square': 'square'}

    fig = go.Figure()

    # Add scatter plot points with customizations
    for shape in df['shape_label'].unique():
        filtered_df = df[df['shape_label'] == shape]
        fig.add_trace(go.Scatter(
            x=filtered_df['x'],
            y=filtered_df['y'],
            mode='markers',
            text=filtered_df['label'],  # Adding point labels
            marker=dict(
                size=10,
                symbol=shape_map[shape],
                color=filtered_df['color_label'].map({'Category 1': 'rgba(93, 164, 214, 0.8)', 'Category 2': 'rgba(255, 144, 14, 0.8)'}),
                opacity=filtered_df['opacity_label'],
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name=shape
        ))

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
    line_x = np.array([df['x'].min(), df['x'].max()])
    line_y = slope * line_x + intercept
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Regression Line'
    ))

    # Add regression statistics
    r_squared = r_value**2
    pearson_corr, _ = stats.pearsonr(df['x'], df['y'])
    fig.add_annotation(
        text=f"R²: {r_squared:.2f}<br>Pearson: {pearson_corr:.2f}",
        xref="paper", yref="paper",
        x=0.05, y=0.95, showarrow=False,
        bordercolor="black", borderwidth=1
    )

    # Adding annotations
    fig.add_annotation(
        text="Sample Annotation",  # Arbitrary text annotation
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )

    # Adding CBBI logo watermark as an example (replace with actual graphic location)
    fig.add_layout_image(
        dict(
            source="https://example.com/logo.png",  # Placeholder image
            xref="paper", yref="paper",
            x=0.5, y=0.5,  # Positioning
            sizex=0.2, sizey=0.2,
            xanchor="center", yanchor="middle",
            opacity=0.3,
            layer="below"
        )
    )

    # Applying global configuration
    layout = set_global_plotly_config()
    fig.update_layout(layout)

    return fig

# Create and display a 2D scatter plot
fig_2d = create_2d_scatter(df)
fig_2d.show()


# # 3D Scatter plot

# In[13]:


# 3D Scatter plot
def set_global_plotly_config():
    """
    Set global plotly configuration for consistent styling.
    
    Returns:
        layout (dict): Global layout configuration for plotly figures.
    """
    layout = dict(
        title="3D Scatter Plot with Advanced Features",
        scene=dict(
            xaxis=dict(title='X Axis'),
            yaxis=dict(title='Y Axis'),
            zaxis=dict(title='Z Axis')
        ),
        plot_bgcolor='rgba(245, 245, 245, 1)',
        showlegend=False,
    )
    return layout


def create_3d_scatter(df):
    """
    Create a 3D scatter plot with basic customization options.
    
    Args:
        df (pd.DataFrame): Input data frame in long or wide format.
    
    Returns:
        fig (plotly.graph_objs.Figure): The configured 3D scatter plot.
    """
    fig = go.Figure()

    # Add 3D scatter plot points
    fig.add_trace(go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        text=df['label'],  # Adding point labels
        marker=dict(
            size=5,
            color=df['z'],  # Color scale based on 'z' value
            colorscale='Viridis',
            opacity=0.8
        )
    ))
    
    # Adding annotations
    fig.add_annotation(
        text="Sample Annotation",  # Arbitrary text annotation
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )

    # Adding CBBI logo watermark as an example (replace with actual graphic location)
    fig.add_layout_image(
        dict(
            source="https://example.com/logo.png",  # Placeholder image
            xref="paper", yref="paper",
            x=0.5, y=0.5,  # Positioning
            sizex=0.2, sizey=0.2,
            xanchor="center", yanchor="middle",
            opacity=0.3,
            layer="below"
        )
    )
    
    # Applying global configuration
    layout = set_global_plotly_config()
    fig.update_layout(layout)

    return fig

# Create and display a 3D scatter plot
fig_3d = create_3d_scatter(df)
fig_3d.show()


# # Line plot

# ### Parameters for the data

# In[14]:


# Parameters for the data
np.random.seed(0)
n_points = 365  # Number of data points (e.g., daily data for a year)
trend = np.linspace(0, 50, n_points)  # Linear upward trend
seasonality = 25 * np.sin(np.linspace(0, 3 * np.pi, n_points))  # Seasonal component
noise = np.random.normal(scale=5, size=n_points)  # Random noise

# Generate data
data = trend + seasonality + noise
x = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
df = pd.DataFrame({'x': x, 'y': data})


# In[15]:


# Error calculation
df['error'] = np.random.uniform(2, 5, size=n_points)  # Example error values


# In[16]:


# Global configuration for the plot layout
def set_global_plotly_config():
    layout = {
        'font': {
            'color': 'black',
            'size': 14
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'legend': {
            'x': 1,
            'y': 1,
            'xanchor': 'right',
            'yanchor': 'top',
            'borderwidth': 1
        },
        'colorway': ['#006E96', '#00263D', '#B8D87A', '#8E847A', '#EB6852', '#983E53'],
    }
    return layout

# Line plot with options
def create_line_plot(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='lines',
        line=dict(
            color='blue',
            width=2,
            dash='solid'
        ),
        opacity=0.8,
        name='Main Line'
    ))

    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'] + df['error'],
        mode='lines',
        line=dict(color='rgba(0,100,80,0.2)', width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'] - df['error'],
        mode='lines',
        fill='tonexty',
        line=dict(color='rgba(0,100,80,0.2)', width=0),
        showlegend=False
    ))

    smoothed_y = gaussian_filter1d(df['y'], sigma=2)
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=smoothed_y,
        mode='lines',
        line=dict(color='#EB6852', width=2, dash='dash'),
        name='Smoothed Line'
    ))
    
    # Adding annotations
    fig.add_annotation(
        text="Sample Annotation",  # Arbitrary text annotation
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False
    )

    # Adding CBBI logo watermark as an example (replace with actual graphic location)
    fig.add_layout_image(
        dict(
            source="https://example.com/logo.png",  # Placeholder image
            xref="paper", yref="paper",
            x=0.5, y=0.5,  # Positioning
            sizex=0.2, sizey=0.2,
            xanchor="center", yanchor="middle",
            opacity=0.3,
            layer="below"
        )
    )
    # Applying global configuration
    layout = set_global_plotly_config()
    fig.update_layout(layout)

    return fig

# Create and display a line plot
fig_line = create_line_plot(df)
fig_line.show()


# In[ ]:




