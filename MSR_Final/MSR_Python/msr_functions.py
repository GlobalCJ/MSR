import pandas as pd
import numpy as np
from statistics import pstdev
import matplotlib.pyplot as plt
import seaborn as sns


import glob
import random
import base64

from PIL import Image
from io import BytesIO
from IPython.display import HTML


print("Hi, welcome to MSR-Functions. Created on 12 November 2021.\n")

print("The library contains the functions below: ")
print("1. dstudent_creator(df)")
print("2. dscourse_creator(df)")
print("3. visual_dist_marks(df)")
print("4. visual_dist_marks_percentile(df)")
print("5. visual_grade_student(column, df)")
print("6. visual_grade_course(column, df)")
print("7. visual_sparkline_bar(df)")


# function to create student df with calculated columns (mean, stddev, freqeuncy etc)
# It accepts df that has std id and 6 courses as columns
def dfstudent_creator(df):

    dfstudent = df.copy()
    # #############CUMULATIVE FREQUENCY#######################################
    # Creating a list of grades
    grades = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dfstudent["grade"] = grades

    # List for CUMULATIVE count
    clist = []
    for i in dfstudent["grade"]:
        x = dfstudent[
            dfstudent[
                ["Course1", "Course2", "Course3", "Course4", "Course5", "Course6"]
            ]
            <= i
        ].count()
        c = 0
        for j in x:
            c = c + j
        clist.append(c)

    # List for FREQUENCY count
    flist = []
    for i in dfstudent["grade"]:
        x = dfstudent[
            dfstudent[
                ["Course1", "Course2", "Course3", "Course4", "Course5", "Course6"]
            ]
            == i
        ].count()
        c = 0
        for j in x:
            c = c + j
        flist.append(c)

    # Cumulative % list (<=)
    c_perc_list = []
    count = (
        dfstudent[["Course1", "Course2", "Course3", "Course4", "Course5", "Course6"]]
        .count()
        .sum()
        * 100
    )

    for i in dfstudent["grade"]:
        x = (
            dfstudent[
                dfstudent[
                    ["Course1", "Course2", "Course3", "Course4", "Course5", "Course6"]
                ]
                <= i
            ]
            .count()
            .sum()
        )
        y = round((x / count) * 10000, 1)
        c_perc_list.append(y)

    # Frequency % list (==)
    f_perc_list = []
    count = (
        dfstudent[["Course1", "Course2", "Course3", "Course4", "Course5", "Course6"]]
        .count()
        .sum()
        * 100
    )

    for i in dfstudent["grade"]:
        x = (
            dfstudent[
                dfstudent[
                    ["Course1", "Course2", "Course3", "Course4", "Course5", "Course6"]
                ]
                == i
            ]
            .count()
            .sum()
        )
        y = round((x / count) * 10000, 1)
        f_perc_list.append(y)

    # ############################ SUM,MEAN,STD,SKEW,KURTOSIS####################

    # Sum of all grades of each Student
    sumS_list = []
    for s, i, j, k, x, y, z in df.itertuples(index=False):

        sumS_list.append(sum([i, j, k, x, y, z]))

    # Mean of all grades of each Student
    meanS_list = round(df.mean(axis=1), 2)

    # Std_Dev of all grades of each Student
    stdS_list = []
    for s, i, j, k, x, y, z in df.itertuples(index=False):
        # Calculating STD_Dev for Population
        stdS_list.append(round(pstdev([i, j, k, x, y, z]), 2))

    # Skewness of all grades of each Student
    skewS_list = round(df.skew(axis=1), 2)

    # Kurtosis of all grades of each Student
    kurtS_list = round(df.kurtosis(axis=1), 2)

    # ########################## ADDING THE LISTS INTO DFSTUDENT ######################

    # Cumulative & Frequency
    dfstudent["Cumulative"] = clist
    dfstudent["Frequency"] = flist
    dfstudent["Cumulative%"] = c_perc_list
    dfstudent["Frequency%"] = f_perc_list

    # Sum, Mean, Std_Dev, Skewness and Kurtosis
    dfstudent["Sum"] = sumS_list
    dfstudent["Mean"] = meanS_list
    dfstudent["Std_Dev"] = stdS_list
    dfstudent["Skewness"] = skewS_list
    dfstudent["Kurtosis"] = kurtS_list
    dfstudent["Average_Mean"] = round(dfstudent["Mean"].mean(), 2)
    dfstudent["Average_Std_Dev"] = round(dfstudent["Std_Dev"].mean(), 2)
    dfstudent["Average_Skewness"] = round(dfstudent["Skewness"].mean(), 2)
    dfstudent["Average_Kurtosis"] = round(dfstudent["Kurtosis"].mean(), 2)

    print("DataFrame with new coumns is created!")

    return dfstudent


# Function below transposes the table and carry our the same functions as above for Courses ISO students
def dfcourse_creator(df):
    dfT = df
    dfT = dfT.set_index("Students")
    dfT = dfT.T
    column_names = dfT.columns
    dfT

    dfcourses = pd.DataFrame()

    lisst = []
    for i in dfT.index:
        lisst.append(i)

    dfcourses["Course"] = lisst

    lisst = []
    for x in column_names:
        item = dfT[x].values
        lisst.append(item)
        dfcourses[x] = item

        # Sum of all grades of each Course
    sumC_list = dfcourses.sum(axis=1)

    # Mean of all grades of each Course
    meanC_list = round(dfcourses.mean(axis=1), 2)

    # Std_Dev of all grades of each Course
    stdC_list = []
    for s, a, b, c, d, e, f, g, h, i, j, k in dfcourses.itertuples(index=False):
        # Calculating STD_Dev for Population
        stdC_list.append(round(pstdev([a, b, c, d, e, f, g, h, i, j, k]), 2))

    # Skew of all grades of each Course
    skewC_list = round(dfcourses.skew(axis=1), 2)

    # Kurtosis of all grades of each Course
    kurtC_list = round(dfcourses.kurtosis(axis=1), 2)
    # Adding the lists above into dfcourses

    dfcourses["Sum"] = sumC_list
    dfcourses["Mean"] = meanC_list
    dfcourses["Std_Dev"] = stdC_list
    dfcourses["Skewness"] = skewC_list
    dfcourses["Kurtosis"] = kurtC_list

    dfcourses["Average_Mean"] = round(dfcourses.Mean.mean(), 2)
    dfcourses["Average_Std_Dev"] = round(dfcourses.Std_Dev.mean(), 2)
    dfcourses["Average_Skewness"] = round(dfcourses.Skewness.mean(), 2)
    dfcourses["Average_Kurtosis"] = round(dfcourses.Kurtosis.mean(), 2)

    print("DataFrame with new coumns is created!")

    return dfcourses


# #######Visualizations #########


def visual_dist_marks(dfstudent):
    ############### Frequency & Cumulative ################
    # Subplot
    fig, ax1 = plt.subplots(figsize=(13, 6))
    # Barplot
    ax1.set_title("Overall Distribution of Marks", fontsize=16)
    ax1.set_xlabel("Marks", fontsize=16)
    ax1.set_ylabel("Frequency in number", fontsize=16)
    ax1 = sns.barplot(x="grade", y="Frequency", data=dfstudent, color="#332c2b")
    ax1.tick_params(axis="y")
    # Creating another axis
    ax2 = ax1.twinx()
    # Lineplot
    ax2 = sns.lineplot(
        x="grade", y="Cumulative", data=dfstudent, color="#e06704", linewidth=3.5
    )
    ax2.set_ylabel("Cumulative", fontsize=16)

    ax2.tick_params(axis="y")


def visual_dist_marks_percentile(dfstudent):
    ############### Frequency% & Cumulative% ################
    # Subplot
    fig, ax1 = plt.subplots(figsize=(13, 6))
    # Barplot
    ax1.set_title("Overall Distribution of Marks (%)", fontsize=16)
    ax1.set_xlabel("Marks", fontsize=16)
    ax1.set_ylabel("Frequency in number", fontsize=16)
    ax1 = sns.barplot(x="grade", y="Frequency%", data=dfstudent, color="#332c2b")
    ax1.tick_params(axis="y")
    # Creating another axis
    ax2 = ax1.twinx()
    # Lineplot
    ax2 = sns.lineplot(
        x="grade", y="Cumulative%", data=dfstudent, color="#e06704", linewidth=3.5
    )
    ax2.set_ylabel("Cumulative%", fontsize=16)
    ax2.tick_params(axis="y")


# Function for Student Visuals
def visual_grade_student(column, df):
    column = column
    dfstudent = df
    fig, ax1 = plt.subplots(figsize=(15, 6))
    sns.set_style("darkgrid")

    ax1.set_title((column + " of Grade by Students"), fontsize=16)
    ax1.set_xlabel("Students", fontsize=16)
    ax1.set_ylabel((column + " Grade"), fontsize=16)
    sns.lineplot(
        y=column,
        x="Students",
        data=dfstudent,
        ci=None,
        marker="o",
        color="#e06704",
        label=(column + " of Grades by Student"),
        linewidth=5,
    )
    x = sns.lineplot(
        y=("Average_" + column),
        x="Students",
        data=dfstudent,
        color="#332c2b",
        label=(column + " of Grades"),
        linewidth=5,
    )
    x.lines[1].set_linestyle("--")


# Function for Course Visuals
def visual_grade_course(column, df):
    dfcourses = df
    column = column
    fig, ax1 = plt.subplots(figsize=(15, 6))
    sns.set_style("darkgrid")

    ax1.set_title((column + " of Grade by Course"), fontsize=16)
    ax1.set_xlabel("Courses", fontsize=16)
    ax1.set_ylabel((column + " Grade"), fontsize=16)
    sns.lineplot(
        y=column,
        x="Course",
        data=dfcourses,
        ci=None,
        marker="o",
        color="#e06704",
        label=(column + " of Grades by Course"),
        linewidth=5,
    )
    x = sns.lineplot(
        y=("Average_" + column),
        x="Course",
        data=dfcourses,
        color="#332c2b",
        label=(column + " of Grades"),
        linewidth=5,
    )
    x.lines[1].set_linestyle("--")


################### SPARKLINESSS and BAR VISUALS IN DF #########################
###############################################################################


def bar_inline_s(data, figsize=(0.5, 1)):
    d = ["a", "b", "c", "d", "e", "f"]

    fig, ax = plt.subplots(1, 1, figsize=(2.8, 0.7))
    ax = sns.barplot(x=d, y=data, color="#ff0000")
    for k, v in ax.spines.items():
        v.set_visible(False)

    sns.set(rc={"axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"})

    ax.set_xticks([])
    ax.set_yticks([])
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(
        base64.b64encode(img.read()).decode()
    )


def bar_inline(data, figsize=(4, 0.25)):
    d = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

    fig, ax = plt.subplots(1, 1, figsize=(2.8, 1))
    ax = sns.barplot(x=d, y=data, color="#ff0000")
    for k, v in ax.spines.items():
        v.set_visible(False)
    sns.set(rc={"axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"})
    ax.set_xticks([])
    ax.set_yticks([])
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(
        base64.b64encode(img.read()).decode()
    )


def sp_inline_s(data, figsize=(0.5, 1)):
    d = ["a", "b", "c", "d", "e", "f"]

    fig, ax = plt.subplots(1, 1, figsize=(2.8, 0.7))
    ax = sns.lineplot(x=d, y=data, color="#ff0000", linewidth=3)
    for k, v in ax.spines.items():
        v.set_visible(False)
    sns.set(rc={"axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"})

    ax.set_xticks([])
    ax.set_yticks([])
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(
        base64.b64encode(img.read()).decode()
    )


def sp_inline(data, figsize=(4, 0.25)):
    d = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

    fig, ax = plt.subplots(1, 1, figsize=(2.8, 1))
    ax = sns.lineplot(x=d, y=data, color="#ff0000", linewidth=3)
    for k, v in ax.spines.items():
        v.set_visible(False)
    sns.set(rc={"axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"})
    ax.set_xticks([])
    ax.set_yticks([])
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    plt.close()
    return '<img src="data:image/png;base64,{}"/>'.format(
        base64.b64encode(img.read()).decode()
    )


def visual_sparkline_bar(df):
    test = df.T
    test.columns = test.iloc[0]
    test = test[1:]

    hh = []
    for i, row in test.iterrows():
        hh.append(row)

    hist_list = []
    score_list = []

    list(hh[0])
    for i in range(6):
        x = list(hh[i])
        x = bar_inline(x)
        hist_list.append(x)

        x = list(hh[i])
        x = sp_inline(x)
        score_list.append(x)

    test["Histogram"] = hist_list
    test["Scores"] = score_list

    test = test.T

    # Creating the Columns with Inline plot

    hh = []
    for i, row in test.iterrows():
        hh.append(row)

    hh

    hhh = []
    for i in range(11):
        hhh.append(hh[i])

    hist_list = []
    score_list = []
    list(hhh[0][:6])

    for i in range(11):
        x = list(hhh[i][:6])
        x = bar_inline_s(x)
        hist_list.append(x)

        x = list(hhh[i][:6])
        x = sp_inline_s(x)
        score_list.append(x)

    hist_list.append(" ")
    hist_list.append(" ")
    score_list.append(" ")
    score_list.append(" ")
    test["Histogram"] = hist_list
    test["Scores"] = score_list
    HTML(
        """<style>
                    .right_aligned_df td { text-align: right; }
                    .left_aligned_df td { text-align: right; }
                    .pink_df { background-color: #ffc4c4; }
                </style>"""
    )

    return HTML(test.to_html(classes="pink_df", escape=False))
