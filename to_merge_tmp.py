from __future__ import print_function
from __future__ import division
import sys
import pandas as pd


def main(argv=None):
    days = int(argv[1])
    df1 = pd.read_csv(f"data/interpolated_D/interpolated_disaster_spx_{days}.csv")
    df2 = pd.read_csv(f"data/interpolated_D/interpolated_disaster_spx_VIV_{days}.csv")
    pd.concat(
        [df1,df2], ignore_index=True
    ).to_csv(
        f"data/interpolated_D/interpolated_disaster_spx_final_{days}.csv",
        index=False
    )

    df1 = pd.read_csv(f"data/interpolated_D/interpolated_disaster_individual_{days}.csv")
    df2 = pd.read_csv(f"data/interpolated_D/interpolated_disaster_individual_VIV_{days}.csv")
    pd.concat(
        [df1,df2], ignore_index=True
    ).to_csv(
        f"data/interpolated_D/interpolated_disaster_individual_final_{days}.csv",
        index=False
    )

if __name__ == "__main__":
    sys.exit(main(sys.argv))
