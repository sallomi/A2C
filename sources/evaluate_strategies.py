
import pandas as pd
import numpy as np
from utils import arp_measure, gini_measure
from PoiEnv import poi_env
from poi_itinerary_with_a2c_tf import evaluate_a2c

def evaluate_strategies(df_poi_it, df_crowding, df_weather_test, df_poi_test, df_poi_time_travel, mode_label,
                        itinerary_dqn, itinerary_a2c, itinerary_h, itinerary_dp, itinerary_cdp):
    strategies = {
        'D-RL': itinerary_dqn,
        'A2C': itinerary_a2c,
        'B-H': itinerary_h,
        'B-DP': itinerary_dp,
        'B-CDP': itinerary_cdp
    }

    list_poi = list(df_poi_it['id'].values)
    summary_table = []

    for strategy_name, df in strategies.items():
        if df.empty:
            continue

        for time_input in sorted(df['time_input'].unique()):
            if time_input not in [4, 5, 6, 7, 8, 9]:
                continue

            group = df[df['time_input'] == time_input]
            if len(group) == 0:
                continue

            total_time = time_input * 60
            vt = (group['time_visit'].mean() / total_time) * 100
            mt = (group['time_distance'].mean() / total_time) * 100
            qt = (group['time_crowd'].mean() / total_time) * 100
            rt = (group['time_left'].mean() / total_time) * 100
            rw = group['reward'].mean()
            arp = arp_measure(list_poi, list(group['itinerary'].values))
            g = gini_measure(list_poi, list(group['itinerary'].values))

            summary_table.append({
                'Visit Time (h)': time_input,
                'Strategy': strategy_name,
                'VT (%)': round(vt, 2),
                'MT (%)': round(mt, 2),
                'QT (%)': round(qt, 2),
                'RT (%)': round(rt, 2),
                'RW': round(rw, 2),
                'ARP': round(arp, 2),
                'G': round(g, 2),
                'Mode': mode_label
            })

    return pd.DataFrame(summary_table)
