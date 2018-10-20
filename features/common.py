import pandas as pd


def aggregate(lc, col, agg, groupby=['object_id','passband']):
    aggs = lc.groupby(groupby)[col].agg(agg)
    aggs.columns = [e[1] + '(' + e[0] + ')' for e in aggs.columns]
    aggs = aggs.unstack()
    aggs.columns = [e[0] + '_ch' + e[1] for e in aggs.columns]
    aggs.reset_index(drop=True, inplace=True)
    return aggs

