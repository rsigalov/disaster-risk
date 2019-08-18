from __future__ import print_function
from __future__ import division
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import wrds
import time

def main(argv = None):

	# User specified properties:
    parser = OptionParser()

    def get_comma_separated_args(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser.add_option('-s','--secidlist', type='string', action='callback',
                      callback=get_comma_separated_args,
                      dest='secid_list',
                      help = """List of SECID to download option data on (e.g. 110012,106445,101966)""")
    parser.add_option("-o", "--output", action="store",
		type="string", dest="output_file",
		help = "Specify output appendix that will be added to opt_data_<appendix>.csv and dist_data_<appendix>.csv")
    parser.add_option("-b", "--startyear",
		type="int", dest="start_year",
		help = "Download data starting from this year")
    parser.add_option("-e", "--endyear",
		type="int", dest="end_year",
		help = "Download data starting up to this year")

    (options, args) = parser.parse_args()
    secid_list = [str(x) for x in options.secid_list]
    index_to_save = options.output_file
    year_start = int(options.start_year)
    year_end = int(options.end_year)

    db = wrds.Connection(wrds_username = "rsigalov")

    # Run a script for each year and get all data for a given company:
    year_list = list(range(year_start, year_end + 1, 1))

    df_prices = pd.DataFrame({"secid": [], "date": [], "exdate": [], "cp_flag": [],
                              "strike_price":[],"impl_volatility":[],"mid_price":[],
                              "under_price":[]})
    num_secid = len(secid_list)
    num_years = len(year_list)

    i_secid = 0

    print("")
    print("--- Start loading option data ---")
    print("")
    start = time.time()
    for secid in secid_list:
        i_secid += 1
        i_year = 0
        secid = str(secid)

        for year in year_list:
            i_year += 1
            print("Secid %s, %s/%s. Year %s, %s/%s" % (str(secid), str(i_secid),
                                                       str(num_secid), str(year),
                                                       str(i_year), str(num_years)))

            start_date = "'" + str(year) + "-01-01'"
            end_date = "'" + str(year) + "-12-31'"
            data_base = "OPTIONM.OPPRCD" + str(year)

            f = open("load_option_list.sql", "r")
            query = f.read()

            query = query.replace('\n', ' ').replace('\t', ' ')
            query = query.replace('_start_date_', start_date).replace('_end_date_', end_date)
            query = query.replace('_secid_', secid)
            query = query.replace('_data_base_', data_base)

            df_option_i = db.raw_sql(query)
            df_prices = df_prices.append(df_option_i)

    end = time.time()
    print("")
    print("--- Time to load option data ---")
    print(end - start)
    print("")

    start = time.time()
    print("")
    print("--- Saving option data ---")
    print("")
    path_to_save_data = "data/raw_data/opt_data_" + index_to_save + ".csv"
    df_prices.to_csv(path_to_save_data, index = False)

    end = time.time()
    print("")
    print("--- Time to save option data ---")
    print(end - start)
    print("")

    print("")
    print("--- Start Loading Distributions Data ---")
    print("")

    # Loading data on index dividend (and other) yield:
    sec_list = "(" + ", ".join(secid_list) + ")"

    query = """
        select *
        from IDXDVD
        where secid in _secid_list_
    """

    query = query.replace('\n', ' ').replace('\t', ' ')
    query = query.replace('_secid_list_', sec_list)
    dist_data = db.raw_sql(query)

    print("")
    print("--- Saving Distributions Data ---")
    print("")
    dist_data.to_csv("data/raw_data/div_yield_" + index_to_save + ".csv", index = False)

    print("")
    print("--- Done! ---")
    print("")


if __name__ == "__main__":
	sys.exit(main(sys.argv))
