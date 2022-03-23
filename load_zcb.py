import numpy as np
import pandas as pd
import psycopg2

def main():
    with open("account_data/wrds_user.txt") as f:
        wrds_username = f.readline()

    with open("account_data/wrds_pass.txt") as f:
        wrds_password = f.readline()

    conn = psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port = 9737,
        database="wrds",
        user=wrds_username,
        password=wrds_password)

    # Downloading interest rates for all dates and all maturity:
    query_zcb = """
    select *
    from OPTIONM.ZEROCD
    where days <= 365*3
    """

    query_zcb = query_zcb.replace('\n', ' ').replace('\t', ' ')
    df_zcb = pd.read_sql_query(query_zcb, conn)

    conn.close()

    df_zcb.to_csv("data/raw_data/zcb_data.csv", index = False)

if __name__ == "__main__":
    main()