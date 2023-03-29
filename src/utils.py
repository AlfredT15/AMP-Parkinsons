# Imports
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import seaborn as sns
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), height=500, width=1000))
color=px.colors.qualitative.Plotly

def preliminary_info(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str):
    print('Number of Training Examples = {}'.format(df_train.shape[0]))
    print('Number of Test Examples = {}\n'.format(df_test.shape[0]))
    print('Training X Shape = {}'.format(df_train.shape))
    print('Training y Shape = {}\n'.format(df_train[target].shape[0]))
    print('Test X Shape = {}'.format(df_test.shape))
    print('Test y Shape = {}\n'.format(df_test.shape[0]))
    print("Target Distribution:\n{}".format((df_train.target.value_counts() / len(df_train)).round(2)))

def print_missing_vals(data: pd.DataFrame, amountMissing: int):
    """Prints out the amount of missing data and the percentages

    :param data: Dataset to find missing data in
    :type data: pd.DataFrame
    :param amountMissing: The amount of features to print out
    :type amountMissing: int
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data[:amountMissing])

def plot_pie_target_distribution(df_train: pd.DataFrame):
    """Plots the target distribution in a pretty pie chart

    :param df_train: Training dataset
    :type df_train: pd.DataFrame
    """
    target=df_train.target.value_counts(normalize=True)[::-1]
    text=['State {}'.format(i) for i in target.index]
    color,pal=['#52489C','#78CAD2'],['#6D63B7','#93D5DB']
    if text[0]=='State 0':
        color,pal=color,pal
    else:
        color,pal=color[::-1],pal[::-1]
    fig=go.Figure()
    fig.add_trace(go.Pie(labels=target.index, values=target*100, hole=.5, 
                        text=text, sort=False, showlegend=False,
                        marker_colors=pal, marker_line=dict(color=color,width=2),
                        hovertemplate = "State %{label}: %{value:.2f}%<extra></extra>"))
    fig.update_layout(template=temp, title='Target Distribution', 
                    uniformtext_minsize=15, uniformtext_mode='hide',width=700)
    fig.show()

def plot_feature_relationships_with_target(df_train: pd.DataFrame, ncols: int):
    float_cols= [cname for cname in df_train.columns if df_train[cname].dtype in ['int64', 'float64']]
    titles=['Feature {}'.format(i.split('_')[-1]) for i in float_cols[:-1]]
    nrows = math.ceil(len(float_cols) // ncols)
    fig=make_subplots(rows=nrows,cols=ncols,
                    subplot_titles=titles,
                    shared_yaxes=True)
    col=np.arange((0,ncols,1),(0,ncols,1), ncols)
    row=0
    pal=sns.color_palette("PuBu",30).as_hex()[12:]
    for i,column in enumerate(float_cols.columns[:-1]):
        if i%ncols==0:
            row+=1
        float_cols['bins'] = pd.cut(float_cols[column],250)
        float_cols['mean'] = float_cols.bins.apply(lambda x: x.mid)
        df = float_cols.groupby('mean')[column,'target'].transform('mean')
        df = df.drop_duplicates(subset=[column]).sort_values(by=column)
        fig.add_trace(go.Scatter(x=df[column], y=df.target, name=column,
                                marker_color=pal[i],showlegend=False),
                    row=row, col=col[i])
        fig.update_xaxes(zeroline=False, row=row, col=col[i])
        if i%4==0:
            fig.update_yaxes(title='Target Probabilitiy',row=row,col=col[i])
    fig.update_layout(template=temp, title='Feature Relationships with Target', 
                    hovermode="x unified",height=1000,width=900)
    fig.show()
    
def plot_correlations_target(df_train: pd.DataFrame):
    corr=df_train.corr().round(2)  
    corr=corr.iloc[:-1,-1].sort_values(ascending=False)
    titles=['Feature '+str(i.split('_')[1]) for i in corr.index]
    corr.index=titles
    pal=sns.color_palette("RdYlBu",32).as_hex()
    pal=[j for i,j in enumerate(pal) if i not in (14,15)]
    rgb=['rgba'+str(matplotlib.colors.to_rgba(i,0.8)) for i in pal] 
    fig=go.Figure()
    fig.add_trace(go.Bar(x=corr.index, y=corr, marker_color=rgb,
                        marker_line=dict(color=pal,width=2),
                        hovertemplate='%{x} correlation with Target = %{y}',
                        showlegend=False, name=''))
    fig.update_layout(template=temp, title='Feature Correlations with Target', 
                    yaxis_title='Correlation', xaxis_tickangle=45, width=800)
    fig.show()

def correlation_coefficient_builder(df_train: pd.DataFrame, df_test: pd.DataFrame, drop_columns: list, ):
    """Creates two dataframes for training set correlation and test set correlation

    :param df_train: Training dataset
    :type df_train: pd.DataFrame
    :param df_test: Testing dataset
    :type df_test: pd.DataFrame
    :param drop_columns: Columns to drop
    :type drop_columns: list
    :return: a tuple of Training dataset and Test dataset
    :rtype: _type_
    """
    df_train_corr = df_train.drop(drop_columns, axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
    df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

    df_test_corr = df_test.drop(drop_columns, axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
    df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
    df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)
    return (df_train_corr_nd ,df_test_corr_nd)

def plot_correlation_heatmap(df_train: pd.DataFrame, drop_columns: list, annotate: bool, df_test: pd.DataFrame = None, figsize: tuple = (20,20)):
    """Plots correlation heat maps

    :param df_train: Training dataset
    :type df_train: pd.DataFrame
    :param df_test: Testing dataset
    :type df_test: pd.DataFrame
    :param drop_columns: Columns to drop
    :type drop_columns: list
    :param annotate: Whether or not to annotate
    :type annotate: bool
    """
    if df_test is not None:
        fig, axs = plt.subplots(nrows=2, figsize=figsize)

        sns.heatmap(df_train.drop(drop_columns, axis=1).corr(), ax=axs[0], annot=annotate, square=True, cmap='coolwarm', annot_kws={'size': 14})
        sns.heatmap(df_test.drop(drop_columns, axis=1).corr(), ax=axs[1], annot=annotate, square=True, cmap='coolwarm', annot_kws={'size': 14})

        for i in range(2):    
            axs[i].tick_params(axis='x', labelsize=14)
            axs[i].tick_params(axis='y', labelsize=14)
            
        axs[0].set_title('Training Set Correlations', size=15)
        axs[1].set_title('Test Set Correlations', size=15)

        plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, figsize=figsize)

        sns.heatmap(df_train.drop(drop_columns, axis=1).corr(), ax=axs, annot=annotate, square=True, cmap='coolwarm', annot_kws={'size': 14})
    
        axs.tick_params(axis='x', labelsize=14)
        axs.tick_params(axis='y', labelsize=14)
        axs.set_title('Training Set Correlations', size=15)

        plt.show()

def plot_feature_distribution(df, feat_list, ncols):
    n_rows = math.ceil(len(feat_list)/ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=n_rows, figsize=(20, 20))
    plt.subplots_adjust(right=1.5)

    for i, feature in enumerate(feat_list):    
        # Distribution of survival in feature
        index1 = i // ncols
        index2 = i % ncols
        sns.distplot(df[feature], label=feature, hist=True, color='#2ecc71', ax=axs[index1][index2])
        
        axs[index1][index2].set_xlabel('')
        
        axs[index1][index2].legend(loc='upper right', prop={'size': 10})
            
    plt.show()


