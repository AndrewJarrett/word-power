import pandas as pd
from pandas import DataFrame, read_sas, read_csv

import statsmodels.api as sm

from SECEdgar.crawler import SecCrawler
from multiprocessing import Pool
from collections import defaultdict
from itertools import repeat
from datetime import datetime as dt
from lxml import etree
from io import StringIO

import os
import re
import time
import string
import pickle
import redis
import zlib
import math
import nltk
from nltk.corpus import stopwords

# Download the stopwords if it has never been done
try: stopwords.words('english')
except LookupError: nltk.download('stopwords')

class WordPower:
    """This class represents the object used for generating Word Power weights. The file main.py will 
    reference this object and call various methods in order to recreate the Word Power results."""

    def __init__(self, start, end):
        """Constructor method for the WordPower class. It takes a start and end date which
        is used to determine what date range and data we should look at when running the 
        analysis. The date range from the paper is 1995 to 2008."""
        self.rds = redis.Redis()
        self.start = start
        self.end = end

        # This regex is used later on, but it is a good idea to compile it once right now rather than
        # every time the function that uses it is called.
        self.filed_regex = re.compile(r"(.*\.sgml\s+?:\s+?|.*\nFILED AS OF DATE:\s+?)([\d]+?)\n.*", re.S)

    # This function reads in relevant data for Word Power
    def load_data(self):
        """This load_data function will check Redis in order to see if data already exists there
        to speed up the process of loading data. Otherwise, it will read data from disk which 
        can take up to about 5 minutes."""

        # Read in SAS data set - takes a while so try to use redis...
        try: self.data
        except AttributeError:
            key = 'data:crsp-comp'
            if self.rds.exists(key):
                print("Loading " + key + " from Redis.")
                self.data = pickle.loads(zlib.decompress(self.rds.get(key)))
            else:
                print("Loading " + key + " from disk.")
                self.data = read_sas("data/crsp_comp.sas7bdat")

                # Trim the SAS data set
                self.data = self.data[['CUSIP','PERMNO','cik','tic','date','PRC','RET','vwretd']]

                # Sort the set by cusip, permno, cik, and then year (descending)
                self.data.sort_values(['CUSIP', 'PERMNO', 'cik', 'date'], ascending=[True, True, True, False], inplace=True)

                # Re-index the dataframe after sorting
                self.data.reset_index(inplace=True, drop=True)

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
        """This is the function that will scrape the SEC Edgar website and download all 10-Ks.
        We are using a third-party package called SECEdgar which does the job, but this would
        be improved if we wrote our own web crawler that could pre-filter any amended 10-Ks and
        10-Ks that fall outside of our date range before downloading them."""
        
        # Remove any duplicates where CUSIP, PERMNO, and CIK match
        ciks = self.data.drop_duplicates(subset=['CUSIP', 'PERMNO', 'cik'])

        # Only keep the cik and ticker column
        ciks = ciks[['cik', 'tic']]

        # Iterate over each CIK and pull the relevant 10k filings
        crawler = SecCrawler()
        end_date = str(self.end) + '1231'
        count = str(math.ceil((self.end - self.start) / 10) * 10)
        
        p = Pool()
        rows = ciks.to_dict(orient='records')
        results = p.starmap(crawl, zip(rows, repeat(end_date), repeat(count), repeat(crawler)))

    # This function will check Redis to see if a file has been processed or cleaned already so we can skip it
    def check_redis(self, cleaned_key, processed_key, report_key):
        """This function takes as input three keys that are used to verify if a 10-K file has been
        cleaned, processed, or if a report already exists. This is to prevent a file from being
        cleaned or processed more than once."""

        processed = False
        cleaned = False
        
        if not self.rds.exists(cleaned_key):
            if not self.rds.exists(processed_key):
                # Temporary check to see if this file has been processed fully
                if self.rds.exists(report_key):
                    mtime = self.rds.hget(report_key, 'mtime')

                    if not self.rds.hexists(report_key, 'company_data'):
                        # Hasn't been cleaned with the new algorithm, so keep booleans False
                        pass
                    elif self.rds.hexists(report_key, 'hist_ret'):
                        processed = True
                        cleaned = True

                        # Save to proper place in redis
                        self.rds.set(cleaned_key, mtime)
                        self.rds.set(processed_key, mtime)
                    else:
                        cleaned = True

                        # Save to proper place in redis
                        self.rds.set(cleaned_key, mtime)
            else:
                processed = True
        else:
            # Check to see if this has really been cleaned (company_data exists)
            if self.rds.hexists(report_key, 'company_data'):
                cleaned = True
                if self.rds.exists(processed_key):
                    processed = True
            
        return (cleaned, processed)

    # This function will move a file to another folder based on if some error occurred.
    def move_file(self, fh, fn, folder, tic, cik, filename, message):
        """This move_file function will move a file into another folder and print a message
        based on if a certain error has occurred. For example, the file can be outside of
        our date range, we can have no stock data for the company on the filing date, or
        it could possibly be too malformed for us to process the file."""

        # Generate the new name of the file
        s = os.sep
        new_name = 'data' + s + folder + s + tic + '-' + cik + '-' + filename

        # Close the file so that we can move it
        fh.close()
        os.rename(fn, new_name)
        print(message)

    # This function handles the cleaning of the 10-K
    def clean(self, fn):
        """This clean function will take a 10-K file, fix some malformed tags, get the filing date,
        strip out any HTML or formatting, check if we should process the 10-K, filter out any words 
        that are not in our 2of12inf dictionary and then store the cleaned up raw text of the report
        in Redis for later use."""

        error = False
        
        s = os.sep # Get the correct folder separator for this OS
        tic = fn.split(s)[1]
        cik = fn.split(s)[2]
        filename = fn.split(s)[4]
        report_key = "report:" + cik + ":" + fn
        cleaned_key = "cleaned:" + cik + ":" + fn

        # Open the file, get all of the content, and then pull it into a parser
        fh = open(fn, 'r')
        contents = fh.read()

        # Clean up some of the text to fix malformed HTML before parsing it
        malformed_tags = ['ACCEPTANCE-DATETIME', 'TYPE', 'SEQUENCE', 'FILENAME', 'DESCRIPTION']
        for tag in malformed_tags:
            # Use a regex that fixes all of these malformed tags in the document
            # TODO: It may be beneficial to find a way to not compile this regex every time
            regex = re.compile(r"(\n<%s>[^<]*?)\n" % re.escape(tag), re.I)
            contents = regex.sub(r"\1</%s>\n" % tag, contents)

        # Create the parser. We use lxml/etree with XPath calls for speed and efficiency
        parser = etree.HTMLParser()
        document = etree.parse(StringIO(contents), parser)
        doc = document.getroot()
        
        # The document can either have a root node of sec-document or ims-document
        if len(doc.xpath('//sec-document')) != 0:
            root = doc.xpath('//sec-document[1]')[0]
        elif len(doc.xpath('//ims-document')) != 0: 
            root = doc.xpath('//ims-document[1]')[0]
        elif len(doc.xpath('//document')) != 0:
            root = doc.xpath('//document[1]')[0]
        elif len(doc.xpath('//error')) != 0:
            root = None
        else:
            root = None
            
        if root is None:
            # Root node error 
            self.move_file(fh, fn, "_error", tic, cik, filename, "No root or erroneous root node - moved file")
            error = True
        if error: return error

        # Check if this is an amended 10-K and throw it out if so
        type_text = root.xpath('//type/text()')
        if type_text is None or len(type_text) == 0:
            self.move_file(fh, fn, "_error", tic, cik, filename, "Error finding type - moved file")
            error = True
        elif type_text[0] == '10-K/A':
            self.move_file(fh, fn, "_amended", tic, cik, filename, "Amended 10-K - moved file")
            error = True
        if error: return error

        # Get the 'acceptance-datetime' metadata element
        acc_dt = root.xpath('//acceptance-datetime/text()')
        if acc_dt is None or len(acc_dt) == 0:
            header_text = None

            # If we didn't find an <acceptance-datetime /> element, find the date elsewhere
            if len(root.xpath('//sec-header/text()')) != 0:
                header_text = root.xpath('//sec-header/text()')[0]
            elif len(root.xpath('//ims-header/text()')) != 0:
                header_text = root.xpath('//ims-header/text()')[0]

            if header_text:
                filing_dt_text = re.sub(self.filed_regex, r"\2", header_text)
            else:
                self.move_file(fh, fn, "_error", tic, cik, filename, "Bad filing date - moved file")
                error = True
            if error: return error
        else:
            # Get the filing date
            filing_dt_text = acc_dt[0].split('\n', 1)[0][:8]

        # Get the Unix timestamp and an actual DateTime object for this filing date
        filing_dt = dt.strptime(filing_dt_text, '%Y%m%d')
        filing_ts = time.mktime(filing_dt.timetuple())
        begin_dt = dt(self.start, 1, 1)

        # If the filing date is not within our date range, then move it
        if begin_dt > filing_dt:
            self.move_file(fh, fn, "_outofrange", tic, cik, filename, "Out of date range - moved file.")
            error = True
        if error: return error

        # See if we can find stock info for this company on the filing date of the 10-K
        index = 0
        cik_df = None
        try:
            index = self.df.index.get_loc((bytes(cik, 'utf-8'), filing_dt))
            cik_df = self.df.ix[bytes(cik, 'utf-8')]
            price = cik_df.ix[filing_dt, 'PRC']

            # Now, check if the price of the stock is less than $3.00
            if price < 3.0:
                self.move_file(fh, fn, "_nostockdata", tic, cik, filename, "Price less than $3.00 - moved file.")
                error = True
        except (IndexError, KeyError):
            # We couldn't find the cik or date for this 10-k
            self.move_file(fh, fn, "_nostockdata", tic, cik, filename, "No stock data found - moved file.")
            error = True
        if error: return error
        
        # Grab the report
        report = ''.join(root.xpath('//document/text')[0].itertext())

        # We will tokenize the text and iterate through each word
        tokens = report.split()
        keep_tokens = []
        stopwords_set = set(stopwords.words('english'))
        punc_table = str.maketrans("", "", string.punctuation)
        
        # Filter out words
        for word in tokens:
            # Quick check to make sure we should keep filtering the word
            if len(word) != 1:
                # Strip punctuation from the word first and make it lowercase
                word = word.translate(punc_table).lower()

                # Add the word to the keep pile if it is not a stopword and if it is in 2of12inf dictionary
                if word not in stopwords_set and word in self.dict_2of12inf:
                    keep_tokens.append(word)
                
        tokens = keep_tokens
        report = " ".join(tokens)
        total_words = len(tokens)

        # Gather info for report to save into redis
        report_hash = {
            'cik': cik,
            'tic': tic,
            'path': fn,
            'file_name': filename,
            'filing_date': filing_ts,
            'year': filing_dt.year,
            'report': report,
            'total_words': total_words,
            'company_data': pickle.dumps(cik_df),
            'index': index,
            'mtime': time.time()
        }

        # Close the file handle
        fh.close()
        
        # Save the stuff to redis
        print("Saving to redis: " + report_key)
        self.rds.hmset(report_key, report_hash)
        self.rds.set(cleaned_key, time.time())

    # This function processes the cleaned report by counting the frequency of positive and negative 
    # words and then calculating the historical and abnormal return based on the 10-K filing date.
    def frequency_returns(self, fn):
        """This frequency_returns function will take a cleaned 10-K file, count the number of times a positive
        or negative word occurs, and then calculate the historical and abnormal return based on
        the 10-Ks filing date. It will then save this info in Redis for later use."""
        
        s = os.sep
        tic = fn.split(s)[1]
        cik = fn.split(s)[2]
        report_key = "report:" + cik + ":" + fn
        processed_key = "processed:"+ cik + ":" + fn
        
        # Get the report out of redis
        report = str(self.rds.hget(report_key, 'report'))
        filing_dt = dt.fromtimestamp(int(float(self.rds.hget(report_key, 'filing_date').decode('utf-8'))))
        cik_df = pickle.loads(self.rds.hget(report_key, 'company_data'))
        index = int(self.rds.hget(report_key, 'index'))
        report_hash = {}

        # Now that everything is cleaned up, we can run the word processing algorithm
        pos_occurs = defaultdict(int)
        neg_occurs = defaultdict(int)
        negators = pd.Series(['not', 'no', 'never'])

        # We will tokenize the text and iterate through each word
        tokens = pd.Series(report.split())

        # Now, process the text
        for i, token in tokens.iteritems():
            if token in self.pos_dict:
                # Check to see if there is a negator
                negated = False
                for word in tokens.iloc[(i - 3):(i + 3)]:
                    if word in negators.values:
                        negated = True
                if not negated:
                    root = self.pos_roots_map[token]
                    pos_occurs[root] += 1
            elif token in self.neg_dict:
                # Check to see if there is a negator
                negated = False
                for word in tokens.iloc[(i - 3):(i + 3)]:
                    if word in negators.values:
                        negated = True
                if not negated:
                    root = self.neg_roots_map[token]
                    neg_occurs[root] += 1

        # For the roots we didn't find, set frequency to zero
        for root in self.pos_roots:
            if root not in pos_occurs:
                pos_occurs[root] = 0
        for root in self.neg_roots:
            if root not in neg_occurs:
                neg_occurs[root] = 0
                
        # Use the index we found earlier to grab the historical info
        hist_returns = cik_df.ix[(index + 1):, 'RET']
        print(hist_returns)

        # Calculate the historical return before the filing date
        hist_ret = 1.0
        for col, series in hist_returns.iteritems():
            if col == 'RET':
                for r in series:
                    if not math.isnan(r):
                        hist_ret *= (r + 1.0)
        hist_ret = hist_ret - 1.0

        # Use the index we found earlier to grab the four day window returns
        returns = cik_df.ix[(index - 3):(index + 1), ['RET','vwretd']]

        # Calculate the abnormal return: r_i = M{t=0, 3} (ret_i,j) - M{t=0,3} (ret_vwi,t)
        ret = 1.0
        ret_vwi = 1.0
        for col, series in returns.iteritems():
            if col == 'RET':
                for r in series:
                    if not math.isnan(r):
                        ret *= (r + 1.0)
            elif col == 'vwretd':
                for r in series:
                    if  not math.isnan(r):
                        ret_vwi *= (r + 1.0)
        ab_ret = ((ret - 1.0) - (ret_vwi - 1.0))

        # Save results of text processing to key in redis
        report_hash['pos_occurs'] = pickle.dumps(pos_occurs)
        report_hash['neg_occurs'] = pickle.dumps(neg_occurs)
        report_hash['hist_ret'] = hist_ret
        report_hash['ab_ret'] = ab_ret
        report_hash['mtime'] = time.time()

        print("Saving to redis: " + report_key)
        self.rds.hmset(report_key, report_hash)
        self.rds.set(processed_key, time.time())

    # This function will actually look through the 10-K files on disk, clean them, and then
    # process them.
    def process_files(self):
        """ This process_files function will iterate over each 10-K file on disk, check Redis to
        make sure the file has not been cleaned or processed, and then clean and process the file."""

        # Change these for testing
        count = 1
        stop = math.inf
        skip_cleaned = True
        skip_processed = True
        process_file = True

        folder = "SEC-Edgar-data"
        for (dirpath, dirnames, filenames) in os.walk(folder, topdown=False):
            for filename in filenames:
                report_hash = {}
                fn = os.sep.join([dirpath, filename])
                
                if filename.endswith('.txt'):
                    if count > stop:
                        break

                    s = os.sep
                    tic = fn.split(s)[1]
                    cik = fn.split(s)[2]
                    
                    # Check redis to see if we have processed or cleaned the report already
                    cleaned_key = "cleaned:" + cik + ":" + fn
                    processed_key = "processed:" + cik + ":" + fn
                    report_key = "report:" + cik + ":" + fn
                    (cleaned, processed) = self.check_redis(cleaned_key, processed_key, report_key)
                    
                    # If the report has been cleaned or we don't want to clean it anyway, skip this step
                    error = False
                    if not cleaned or not skip_cleaned:
                        print("(" + str(count) + ") Cleaning " + fn)
                        error = self.clean(fn)
                        
                        if not process_file and not error:
                            count += 1
                            continue
                    if error: continue
                    
                    # After possibly cleaning, check if we should process the file (get the frequency of pos/neg 
                    # words and the returns based on the filing date)
                    if (not processed or not skip_processed) and process_file:
                        print("(" + str(count) + ") Processing " + fn)
                        self.frequency_returns(fn)
                        
                        count += 1

    # This function will generate a variable that contains aggregated data for 10-K reports
    # over the year range that we are looking at.
    def generate_yearly_data(self):
        """This generate_yearly_data function will generate a variable that contains aggregated
        data for every 10-K report within Redis over the year range that we want to analyze."""

        # This will hold the aggregated 10-K data where the key is the year
        self.yearly_data = {}

        if self.rds.exists("yearly-data"):
            print("Found yearly data in Redis.")
            self.yearly_data = pickle.loads(zlib.decompress(self.rds.get("yearly-data")))
        else:
            keys = self.rds.keys("report:*")
            errors = []

            for key in keys:
                report_hash = self.rds.hgetall(key)
                try:
                    cik = str(report_hash[b'cik'].decode('utf-8'))
                    fn = str(report_hash[b'path'].decode('utf-8'))
                    key = "report:" + cik + ":" + fn
                    pos_occurs = pickle.loads(report_hash[b'pos_occurs'])
                    neg_occurs = pickle.loads(report_hash[b'neg_occurs'])
                    year = int(report_hash[b'year'])
                    total_words = int(report_hash[b'total_words'])
                    hist_ret = float(report_hash[b'hist_ret'])
                    ab_ret = float(report_hash[b'ab_ret'])

                    # Check if the year for this report is within our date range that we are looking at
                    if self.start <= year <= self.end:
                        if total_words == 0:
                            # Not sure why this would be zero, but we may need to reprocess
                            print("Error with: " + key)
                            cleaned_key = "cleaned:" + cik + ":" + fn
                            processed_key = "processed:" + cik + ":" + fn

                            # Delete this error from redis
                            self.rds.delete(cleaned_key, processed_key, key)

                            errors.append(key)
                            continue

                        try: self.yearly_data[year]
                        except KeyError: self.yearly_data[year] = []

                        year_list = self.yearly_data[year]
                        year_list.append({
                            'pos_occurs': pos_occurs,
                            'neg_occurs': neg_occurs,
                            'total_words': total_words,
                            'hist_ret': hist_ret,
                            'ab_ret': ab_ret
                        })
                        self.yearly_data[year] = year_list
                    else:
                        continue
                except KeyError: continue
                except e: print(e)
                    
            print("Saving yearly data in Redis.")
            self.rds.set("yearly-data", zlib.compress(pickle.dumps(self.yearly_data)))
            
            # See if we encountered any errors
            if len(errors) > 0:
                print("Total errors: " + str(len(errors)))
        
        # Print out a listing of how many 10-Ks are within each year
        for year in sorted(self.yearly_data.keys()):
            print(year, len(self.yearly_data[year]))

    # This function performs the regression analysis
    def regression_analysis(self):
        """The regression_analysis function will iterate through the yearly_data variable
        and run regressions for each year from the start year until the end year."""

        # This will skip the regression analysis or building dataframe if it is
        # already stored in Redis.
        skip_regressions = True
        skip_building = True

        # Generate the yearly_data variable if it doesn't exist
        try: self.yearly_data
        except AttributeError: self.generate_yearly_data()

        # Generate a rolling training model using data up until year T-1
        for t in range((self.start + 1), (self.end + 2)):
            
            print("Predicting year " + str(t))
            if self.rds.exists("regression:" + str(t)) and skip_regressions:
                continue
            
            pos_word_weights = pd.DataFrame()
            neg_word_weights = pd.DataFrame()
            hist_returns = pd.DataFrame()
            ab_returns = pd.DataFrame()
            
            # Iterate over each year before year T and build the training data set
            for year in range((self.start), t):

                key = "regression-data:" + str(year)
                if self.rds.exists(key) and skip_building:
                    # Extract the data
                    pos_word_weights = pickle.loads(zlib.decompress(self.rds.hget(key, 'pos_word_weights')))
                    neg_word_weights = pickle.loads(zlib.decompress(self.rds.hget(key, 'neg_word_weights')))
                    hist_returns = pickle.loads(zlib.decompress(self.rds.hget(key, 'hist_returns')))
                    ab_returns = pickle.loads(zlib.decompress(self.rds.hget(key, 'ab_returns')))
                    continue

                print("Building year " + str(year))
                
                try: self.yearly_data[year]
                except KeyError:
                    print("Year " + str(year) + " not found.")
                    continue
                
                # Parallel process the initial word weights
                p = Pool()
                total = len(self.yearly_data[year]) 
                results = p.starmap(generate_weights, zip(self.yearly_data[year], range(1, total + 1), repeat(total)))
                results_df = pd.DataFrame.from_records(results, columns=['pos_weights', 'neg_weights', 'hist_ret', 'ab_ret'])

                for i in range(0, total):
                    # TODO: Appending rows to a dataframe is not very efficient, so find a better way
                    print("Appending: " + str(i+1) + "/" + str(total))
                    pos_word_weights = pos_word_weights.append(results_df.ix[i]['pos_weights'], ignore_index=True)
                    neg_word_weights = neg_word_weights.append(results_df.ix[i]['neg_weights'], ignore_index=True)
                    hist_returns = hist_returns.append(results_df.ix[i]['hist_ret'], ignore_index=True)
                    ab_returns = ab_returns.append(results_df.ix[i]['ab_ret'], ignore_index=True)

                # Save our progress to redis
                print("Saving progress for year " + str(year) + " to redis.")
                reg_data_hash = {
                    'pos_word_weights': zlib.compress(pickle.dumps(pos_word_weights)),
                    'neg_word_weights': zlib.compress(pickle.dumps(neg_word_weights)),
                    'hist_returns': zlib.compress(pickle.dumps(hist_returns)),
                    'ab_returns': zlib.compress(pickle.dumps(ab_returns))
                }
                self.rds.hmset(key, reg_data_hash)
                
            # Run the regressions for all years up to year T
            if not ab_returns.empty and not hist_returns.empty and not pos_word_weights.empty and not neg_word_weights.empty:
                hist_returns.reset_index()
                hist_returns_series = pd.Series(hist_returns['hist_ret'])
                ab_returns.reset_index()
                ab_returns_series = pd.Series(ab_returns['ab_ret'])
                pos_word_weights.reset_index()
                neg_word_weights.reset_index()
                
                # Estimate the weights for the words using a regression
                print("T = " + str(t) + ": Estimating weights")
                pos_reg = sm.OLS(hist_returns_series, pos_word_weights)
                pos_model = pos_reg.fit()
                neg_reg = sm.OLS(hist_returns_series, neg_word_weights)
                neg_model = neg_reg.fit()
                
                # Map the words to their coefficients
                pos_coeffs_dict = dict(zip(list(pos_word_weights.columns), pos_model.params))
                pos_coeffs = pd.DataFrame(list(pos_coeffs_dict.items()), columns=['word','weight'])
                pos_coeffs.set_index('word', inplace=True)
                neg_coeffs_dict = dict(zip(list(neg_word_weights.columns), neg_model.params))
                neg_coeffs = pd.DataFrame(list(neg_coeffs_dict.items()), columns=['word','weight'])
                neg_coeffs.set_index('word', inplace=True)
                print(hist_returns_series, pos_word_weights)
                #print(pos_model.summary())
            
                # Calculate the average word weight as well as the standard deviation
                pos_avg = pos_coeffs['weight'].mean()
                pos_std = pos_coeffs['weight'].std()
                neg_avg = neg_coeffs['weight'].mean()
                neg_std = neg_coeffs['weight'].std()

                # Normalize the weights of the words
                print("T = " + str(t) + ": Normalizing weights")
                pos_norm = list()
                for col, series in pos_coeffs.iteritems():
                    if col == 'weight':
                        for weight in series:
                            pos_norm.append((weight - pos_avg) / pos_std)
                pos_coeffs['norm_weight'] = pd.Series(pos_norm, index=pos_coeffs.index)
                
                neg_norm = list()
                for col, series in neg_coeffs.iteritems():
                    if col == 'weight':
                        for weight in series:
                            neg_norm.append((weight - neg_avg) / neg_std)
                neg_coeffs['norm_weight'] = pd.Series(neg_norm, index=neg_coeffs.index)
                
                # Iterate through the original word weights and apply the normalized weight
                for word, series in pos_word_weights.iteritems():
                    norm_weight = pos_coeffs.ix[word]['norm_weight']
                    pos_word_weights[word] = series.apply(lambda x: x * norm_weight)
                for word, series in neg_word_weights.iteritems():
                    norm_weight = neg_coeffs.ix[word]['norm_weight']
                    neg_word_weights[word] = series.apply(lambda x: x * norm_weight)
                        
                # Run the regression for abnormal (after filing) returns using the estimated weights for the words
                print("T = " + str(t) + ": Doing regression")
                pos_ab_reg = sm.OLS(ab_returns_series, pos_word_weights)
                pos_ab_model = pos_ab_reg.fit()
                neg_ab_reg = sm.OLS(ab_returns_series, neg_word_weights)
                neg_ab_model = neg_ab_reg.fit()
                
                # Map the words to their coefficients
                pos_coeffs_dict = dict(zip(list(pos_word_weights.columns), pos_ab_model.params))
                pos_coeffs = pd.DataFrame(list(pos_coeffs_dict.items()), columns=['word','weight'])
                pos_coeffs.set_index('word', inplace=True)
                neg_coeffs_dict = dict(zip(list(neg_word_weights.columns), neg_ab_model.params))
                neg_coeffs = pd.DataFrame(list(neg_coeffs_dict.items()), columns=['word','weight'])
                neg_coeffs.set_index('word', inplace=True)
                
                key = "regression:" + str(t)
                reg_hash = {
                    'pos_model': zlib.compress(pickle.dumps(pos_model)),
                    'neg_model': zlib.compress(pickle.dumps(neg_model)),
                    'pos_ab_model': zlib.compress(pickle.dumps(pos_ab_model)),
                    'neg_ab_model': zlib.compress(pickle.dumps(neg_ab_model)),
                    'pos_coeffs': zlib.compress(pickle.dumps(pos_coeffs)),
                    'neg_coeffs': zlib.compress(pickle.dumps(neg_coeffs))
                }
                self.rds.hmset(key, reg_hash)

    # This function will get a count of the number of appearances for each positive or negative word across all 10-Ks.
    def get_appearances(self):
        """The get_appearances function will go through each year of 10-K reports and count
        the number of times a positive or negative word will appear based on the number of
        reports the word occurs in."""

        try: self.yearly_data
        except AttributeError: self.yearly_data = pickle.loads(zlib.decompress(self.rds.get("yearly-data")))

        pos_appearances_dict = defaultdict(int)
        neg_appearances_dict = defaultdict(int)
        pos_app_set = {}
        neg_app_set = {}

        total = 0
        for year in self.yearly_data:
            total += len(self.yearly_data[year])

            i = 0
            for report in self.yearly_data[year]:
                # Keep track if a word appears at least once and increment a counter if so
                pos_occurs = report['pos_occurs']
                neg_occurs = report['neg_occurs']

                for (word, freq) in pos_occurs.items():
                    try: word_set = pos_app_set[word]
                    except: word_set = set()
                        
                    if freq > 0:
                        pos_appearances_dict[word] += 1
                        word_set.add(str(year) + ":" + str(i))
                        pos_app_set[word] = word_set
                    else:
                        pos_appearances_dict[word] += 0
                        pos_app_set[word] = word_set
                        
                for (word, freq) in neg_occurs.items():
                    if freq > 0:
                        neg_appearances_dict[word] += 1
                        word_set.add(str(year) + ":" + str(i))
                        neg_app_set[word] = word_set
                    else:
                        neg_appearances_dict[word] += 0
                        neg_app_set[word] = word_set
                i += 1

        pos_appearances = pd.DataFrame({'apps': pos_appearances_dict, 'files': pos_app_set})
        neg_appearances = pd.DataFrame({'apps': neg_appearances_dict, 'files': neg_app_set})
        return (total, pos_appearances, neg_appearances)

    # This function will run reports
    def run_reports(self):
        """The run_reports function will run reports we want that shows various words with their
        term frequencies as well as their regression coefficient."""

        # This will store the global positive and negative words occurrences
        pos_occurs_all = defaultdict(int)
        neg_occurs_all = defaultdict(int)

        try: self.yearly_data
        except AttributeError: self.yearly_data = pickle.loads(zlib.decompress(self.rds.get('yearly-data')))

        if self.rds.exists("freq:pos:all") and self.rds.exists("freq:neg:all"):
            pos_occurs_all = pickle.loads(zlib.decompress(self.rds.get("freq:pos:all")))
            neg_occurs_all = pickle.loads(zlib.decompress(self.rds.get("freq:neg:all")))
        else:
            for year in self.yearly_data:
                print("Counting frequency for year " + str(year))

                pos_occurs_year = defaultdict(int)
                neg_occurs_year = defaultdict(int)
                pos_key = "freq:pos:" + str(year)
                neg_key = "freq:neg:" + str(year)

                if self.rds.exists(pos_key) and self.rds.exists(neg_key):
                    pos_occurs_year = pickle.loads(zlib.decompress(self.rds.get(pos_key)))
                    neg_occurs_year = pickle.loads(zlib.decompress(self.rds.get(neg_key)))
                else:
                    count = 0
                    total = len(self.yearly_data[year])
                    for report in self.yearly_data[year]:
                        count += 1
                        print("Counting Report " + str(count) + "/" + str(total))
                        report_pos_occurs = report['pos_occurs']
                        report_neg_occurs = report['neg_occurs']

                        for word, freq in report_pos_occurs.items():
                            pos_occurs_year[word] += freq
                        for word, freq in report_neg_occurs.items():
                            neg_occurs_year[word] += freq

                    print("Saving progress to Redis.")
                    self.rds.set("freq:pos:" + str(year), zlib.compress(pickle.dumps(pos_occurs_year)))
                    self.rds.set("freq:neg:" + str(year), zlib.compress(pickle.dumps(neg_occurs_year)))

            for word, freq in pos_occurs_year.items():
                pos_occurs_all[word] += freq
            for word, freq in neg_occurs_year.items():
                neg_occurs_all[word] += freq
                
            print("Finished totaling and now saving to Redis.")
            self.rds.set("freq:pos:all", zlib.compress(pickle.dumps(pos_occurs_all)))
            self.rds.set("freq:neg:all", zlib.compress(pickle.dumps(neg_occurs_all)))

        # Sort the values by highest to lowest
        pos_sorted = pd.Series(data=pos_occurs_all).sort_values()
        neg_sorted = pd.Series(data=neg_occurs_all).sort_values()

        # Get the coefficients
        key = "regression:" + str(self.end + 1)
        pos_coeffs = pickle.loads(zlib.decompress(self.rds.hget(key, 'pos_coeffs')))
        neg_coeffs = pickle.loads(zlib.decompress(self.rds.hget(key, 'neg_coeffs')))

        # Place the weights into quintiles
        pos_wt_buckets = pd.qcut(pos_coeffs['weight'], 5, labels=range(1,6))
        neg_wt_buckets = pd.qcut(neg_coeffs['weight'], 5, labels=range(1,6))

        # Sort the coefficients by value
        pos_coeffs.sort_values('weight', inplace=True)
        neg_coeffs.sort_values('weight', inplace=True)

        # See if we have the total 10-Ks and number of appearances of each word in a 10-k
        try: total, pos_appearances, neg_appearances
        except NameError: (total, pos_appearances, neg_appearances) = self.get_appearances()

        # Place the frequencies into quintiles
        pos_freq_buckets = pd.qcut(pos_appearances['apps'], 5, labels=range(1,6))
        neg_freq_buckets = pd.qcut(neg_appearances['apps'], 5, labels=range(1,6))

        bucket_freq_list = []
        bucket_wt_list = []
        freq_list = []
        total_list = []
        for (word, row) in pos_coeffs.iterrows():
            bucket_freq_list.append(pos_freq_buckets.ix[word])
            bucket_wt_list.append(pos_wt_buckets.ix[word])
            freq_list.append(pos_sorted.ix[word])
            total_list.append(pos_appearances.ix[word]['apps'])
        pos_coeffs['bucket_freq'] = pd.Series(bucket_freq_list, index=pos_coeffs.index)
        pos_coeffs['bucket_weight'] = pd.Series(bucket_wt_list, index=pos_coeffs.index)
        pos_coeffs['freq'] = pd.Series(freq_list, index=pos_coeffs.index)
        pos_coeffs['apps'] = pd.Series(total_list, index=pos_coeffs.index)
        pos_coeffs['files'] = pos_appearances['files']

        bucket_freq_list = []
        bucket_wt_list = []
        freq_list = []
        total_list = []
        for (word, row) in neg_coeffs.iterrows():
            bucket_freq_list.append(neg_freq_buckets.ix[word])
            bucket_wt_list.append(neg_wt_buckets.ix[word])
            freq_list.append(neg_sorted.ix[word])
            total_list.append(neg_appearances.ix[word]['apps'])
        neg_coeffs['bucket_freq'] = pd.Series(bucket_freq_list, index=neg_coeffs.index)
        neg_coeffs['bucket_weight'] = pd.Series(bucket_wt_list, index=neg_coeffs.index)
        neg_coeffs['freq'] = pd.Series(freq_list, index=neg_coeffs.index)
        neg_coeffs['apps'] = pd.Series(total_list, index=neg_coeffs.index)
        neg_coeffs['files'] = neg_appearances['files']

        # Build the table of weight quintiles to frequency quintiles
        pos_tab5 = pos_coeffs.reset_index(level='word').set_index(['bucket_weight','bucket_freq','word'])
        neg_tab5 = neg_coeffs.reset_index(level='word').set_index(['bucket_weight','bucket_freq','word'])

        # Iterate first through the weight buckets
        print("Positive Weight Quintiles by Term Frequency Quintiles")
        for i in range(1,6):
            # Get the total number of words in this weight quantile
            tot_in_quantile = len(pos_tab5.ix[i]['weight'])
            
            print("\nWeight Quintile " + str(i) + ": Total words = " + str(tot_in_quantile))
            
            # Find the number of words in each freq bucket and show that as percentage of words in weight bucket
            for j in range(1,6):
                try: num_words = len(pos_tab5.ix[(i,j)]['weight'])
                # If this index doesn't exist, we set it to zero
                except KeyError: num_words = 0

                perc = (num_words / (1.0 * tot_in_quantile)) * 100
                print("Term Freq Quintile " + str(j) + ": " + str(perc))
                
        print("\nNegative Weight Quintiles by Term Frequency Quintiles")
        for i in range(1,6):
            tot_in_quantile = len(neg_tab5.ix[i]['weight'])
            
            print("\nWeight Quintile " + str(i) + ": Total words = " + str(tot_in_quantile))
            
            for j in range(1,6):
                try: num_words = len(neg_tab5.ix[(i,j)]['weight'])
                # If this index doesn't exist, we set it to zero
                except KeyError: num_words = 0

                perc = (num_words / (1.0 * tot_in_quantile)) * 100
                print("Term Freq Quintile " + str(j) + ": " + str(perc))

        # Now, sort by the frequency buckets (ascending) and weights (descending)
        pos_coeffs.sort_values(['bucket_freq','weight'], ascending=[True,False], inplace=True)
        neg_coeffs.sort_values(['bucket_freq','weight'], ascending=[True,False], inplace=True)

        # Get the top 5 words in each quintile
        pos_top5 = pos_coeffs.reset_index(level=['word']).set_index(['bucket_freq'])
        pos_top5_list = {}
        neg_top5 = neg_coeffs.reset_index(level=['word']).set_index(['bucket_freq'])
        neg_top5_list = {}
        for i in range(1, 6):
            pos_words = []
            pos_bucket_words = pos_top5.ix[i]['word']
            neg_words = []
            neg_bucket_words = neg_top5.ix[i]['word']
            
            count = 0
            for word in pos_bucket_words:
                if count >= 5:
                    break
                pos_words.append(word)
                count += 1
            count = 0
            for word in neg_bucket_words:
                if count >= 5:
                    break
                neg_words.append(word)
                count += 1
            
            pos_top5_list[i] = pos_words
            neg_top5_list[i] = neg_words

        # Print out most frequent positive words
        header = "\n\nTop Five Most Positive and Negative Words within Frequency Quintiles"
        print(header)
        print("="*len(header))

        subheader = "\nTop 5 Most Positive Words"
        print(subheader)
        print("="*len(subheader))

        for (bucket, words) in pos_top5_list.items():
            print("\nBucket " + str(bucket))
            for word in words:
                print(word)

        # Print out most frequent negative words
        subheader = "\nTop 5 Most Negative Words"
        print(subheader)
        print("="*len(subheader))

        for (bucket, words) in neg_top5_list.items():
            print("\nBucket " + str(bucket))
            for word in words:
                print(word)

# This function is for scraping 10-K files and is used via multiprocessing to speed up scraping
def crawl(row, end_date, count, crawler):
    cik = row['cik'].decode('utf-8')
    tic = row['tic'].decode('utf-8')
    crawler.filing_10K(tic, cik, end_date, count)

# Generate the initial weights for each positive and negative word
def generate_weights(report, count, total):
    """This generate_weights function will generate the initial weights for each word
    based on the frequency of the word and the total words within the specific report.
    This function is used within a multiprocessing context to improve performance."""

    print("Report " + str(count) + "/" + str(total))

    a = report['total_words']
    hist_ret = report['hist_ret']
    ab_ret = report['ab_ret']
    
    pos_weights = {}
    pos_occurs = report['pos_occurs']
    for word in pos_occurs.keys():
        F = pos_occurs[word]
        pos_weights[word] = F/(a * 1.0)
    
    neg_weights = {}
    neg_occurs = report['neg_occurs']
    for word in neg_occurs.keys():
        F = neg_occurs[word]
        neg_weights[word] = F/(a * 1.0)

    return (pos_weights, neg_weights, {'hist_ret': hist_ret}, {'ab_ret': ab_ret})
