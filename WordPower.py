import pandas as pd
from pandas import DataFrame, read_sas, read_csv

from SECEdgar.crawler import SecCrawler

import pickle
import redis
import zlib
import math

class WordPower:
    """This class represents the object used for generating Word Power weights"""

    def __init__(self, start, end):
        self.rds = redis.Redis()
        self.start = start
        self.end = end

    # This function reads in relevant data for Word Power
    def load_data(self):
        # Read in SAS data set - takes a while so try to use redis...
        try: self.data
        except AttributeError:
            key = 'data:crsp-comp:' + str(self.start) + '-' + str(self.end)
            if self.rds.exists(key):
                print("Loading " + key + " from Redis.")
                self.data = pickle.loads(zlib.decompress(self.rds.get(key)))
            else:
                print("Loading data:crsp-comp from disk.")
                self.data = read_sas("data/crsp_comp.sas7bdat")

                # Trim the SAS data set
                self.data = self.data[['CUSIP','PERMNO','cik','tic','date','PRC','RET','vwretd']]

                # Sort by date and then drop anything outside the time range

                # Sort the set by cusip, permno, cik, and then year (descending)
                self.data.sort_values(['CUSIP', 'PERMNO', 'cik', 'date'], ascending=[True, True, True, False], inplace=True)

                # Re-index the dataframe after sorting
                self.data.reset_index(inplace=True)

                self.rds.set(key, zlib.compress(pickle.dumps(self.data)))

        # We only need certain columns from the data set and we must set the right index for performance
        try: self.df
        except AttributeError:
            self.df = self.data[["cik", "date", "PRC", "RET", "vwretd"]]
            self.df.set_index(keys=['cik','date'], inplace=True)
                
        # Positive words
        try: self.pos_dict, self.pos_roots, self.pos_roots_map
        except AttributeError:
            if self.rds.exists('data:pos-dict') and self.rds.exists('data:pos-roots') and self.rds.exists('data:pos-roots-map'):
                print("Loading data:pos-dict, data:pos-roots, and data:pos-roots-map from Redis.")
                self.pos_dict = pickle.loads(zlib.decompress(self.rds.get('data:pos-dict')))
                self.pos_roots = pickle.loads(zlib.decompress(self.rds.get('data:pos-roots')))
                self.pos_roots_map = pickle.loads(zlib.decompress(self.rds.get('data:pos-roots-map')))
            else:
                # Read in the positive word list(s)
                print("Loading data:pos-dict, data:pos-roots, and data:pos-roots-map from disk.")
                self.pos_dict = read_csv("data/pos_list.csv", header=None, names=['word'])
                self.pos_dict = set(self.pos_dict['word'])
                self.pos_roots = read_csv("data/pos_roots.csv")
                self.pos_roots_map = dict(zip(list(self.pos_roots.word), list(self.pos_roots.group)))
                self.pos_roots = set(self.pos_roots['group'].drop_duplicates())

                # Save this data to redis for later
                self.rds.set('data:pos-dict', zlib.compress(pickle.dumps(self.pos_dict)))
                self.rds.set('data:pos-roots', zlib.compress(pickle.dumps(self.pos_roots)))
                self.rds.set('data:pos-roots-map', zlib.compress(pickle.dumps(self.pos_roots_map)))

        # Negative words
        try: self.neg_dict, self.neg_roots, self.neg_roots_map
        except AttributeError:
            if self.rds.exists('data:neg-dict') and self.rds.exists('data:neg-roots') and self.rds.exists('data:neg-roots-map'):
                print("Loading data:neg-dict, data:neg-roots, and data:neg-roots-map from Redis.")
                self.neg_dict = pickle.loads(zlib.decompress(self.rds.get('data:neg-dict')))
                self.neg_roots = pickle.loads(zlib.decompress(self.rds.get('data:neg-roots')))
                self.neg_roots_map = pickle.loads(zlib.decompress(self.rds.get('data:neg-roots-map')))
            else:
                # Read in the negative word list(s)
                print("Loading data:neg-dict, data:neg-roots, and data:neg-roots-map from disk.")
                self.neg_dict = read_csv("data/neg_list.csv", header=None, names=['word'])
                self.neg_dict = set(self.neg_dict['word'])
                self.neg_roots = read_csv("data/neg_roots.csv")
                self.neg_roots_map = dict(zip(list(self.neg_roots.word), list(self.neg_roots.group)))
                self.neg_roots = set(self.neg_roots['group'].drop_duplicates())

                # Save this data to redis for later
                self.rds.set('data:neg-dict', zlib.compress(pickle.dumps(self.neg_dict)))
                self.rds.set('data:neg-roots', zlib.compress(pickle.dumps(self.neg_roots)))
                self.rds.set('data:neg-roots-map', zlib.compress(pickle.dumps(self.neg_roots_map)))

        # 2of12inf dictionary
        try: self.dict_2of12inf
        except AttributeError:
            if self.rds.exists('data:2of12inf'):
                print("Loading data:2of12inf from Redis.")
                self.dict_2of12inf = pickle.loads(zlib.decompress(self.rds.get('data:2of12inf')))
            else:
                # Read in the 2of12inf
                print("Loading data:2of12inf from disk.")
                self.dict_2of12inf = read_csv("data/2of12inf.txt", header=None, names=['word'])

                # Iterate through and remove the percent signs
                regex = re.compile(r'%$')
                self.dict_2of12inf.apply(lambda x: re.sub(regex, r'', x['word']), axis=1)
                self.dict_2of12inf = set(self.dict_2of12inf['word'])

                # Save this to redis for later
                self.rds.set('data:2of12inf', zlib.compress(pickle.dumps(self.dict_2of12inf)))

    # This function will scrape the SEC Edgar website for 10-Ks
    def scrape_edgar(self):
        # Remove any duplicates where CUSIP, PERMNO, and CIK match
        ciks = self.data.drop_duplicates(subset=['CUSIP', 'PERMNO', 'cik'])

        # Only keep the cik and ticker column
        ciks = ciks[['cik', 'tic']]

        # Iterate over each CIK and pull the relevant 10k filings
        crawler = SecCrawler()
        end_date = str(self.end) + '1231'
        count = str(math.ceil((self.end - self.start) / 10) * 10)

        print(end_date, count)

        for index, row in ciks.iterrows():
            cik = row.iloc[0]
            tic = row.iloc[1]
            crawler.filing_10K(tic, cik, end_date, count)
