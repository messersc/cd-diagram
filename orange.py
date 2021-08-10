import Orange
import matplotlib.pyplot as plt
import pandas as pd
#import pdb
import sys

def draw_cd_diagram(df_perf, colname):
    df = df_perf.groupby("dataset_name")[colname].rank(ascending=False)
    df["classifier_name"] = df_perf["classifier_name"]
    df = df.groupby("classifier_name").mean()
    n = df_perf["dataset_name"].nunique()
    #print(df)
    #pdb.set_trace()
    avranks = df.values
    names = df.index.tolist()

    cd = Orange.evaluation.compute_CD(avranks, n)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    plt.show()


if __name__ == "__main__":
    print(sys.argv)
    try:
        df = sys.argv[1]
    except:
        df = "example.csv"
    try:
        colname = sys.argv[2]
    except:
        colname = "accuracy"
    df_perf = pd.read_csv(df, index_col=False)
    draw_cd_diagram(df_perf=df_perf, colname=colname)
