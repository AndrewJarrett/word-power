from pandas import DataFrame, read_sas, read_csv
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from SECEdgar.crawler import SecCrawler

from bs4 import BeautifulSoup as bs

import time
from datetime import datetime as dt
from datetime import date, timedelta

from collections import defaultdict
from multiprocessing import Pool
from itertools import repeat
from cpython cimport bool
from lxml import etree
from io import StringIO

import os
import re
import lxml
import redis
import string
import pickle
import math
import zlib

import nltk
from nltk.corpus import stopwords

try: stopwords.words('english')
except LookupError: nltk.download('stopwords')
    
import statsmodels.api as sm
    
rds = redis.Redis()
count = 0
yearly_data = {}

pos_word_weights = pd.DataFrame()
neg_word_weights = pd.DataFrame()
hist_returns = pd.DataFrame()
ab_returns = pd.DataFrame()


def check_redis(str cleaned_key, str processed_key, str report_key):
    cdef bool processed = False
    cdef bool cleaned = False
    cdef str mtime
    
    if not rds.exists(cleaned_key):
        if not rds.exists(processed_key):
            # Temporary check to see if this file has been processed fully
            if rds.exists(report_key):
                mtime = rds.hget(report_key, 'mtime')

                if not rds.hexists(report_key, 'company_data'):
                    # Hasn't been cleaned with the new algorithm, so keep booleans False
                    pass
                elif rds.hexists(report_key, 'hist_ret'):
                    processed = True
                    cleaned = True

                    # Save to proper place in redis
                    rds.set(cleaned_key, mtime)
                    rds.set(processed_key, mtime)
                else:
                    cleaned = True

                    # Save to proper place in redis
                    rds.set(cleaned_key, mtime)
        else:
            processed = True
    else:
        # Check to see if this has really been cleaned (company_data exists)
        if rds.hexists(report_key, 'company_data'):
            cleaned = True
            if rds.exists(processed_key):
                processed = True
        
    return (cleaned, processed)

def move_file(fh, str fn, str folder, str tic, str cik, str filename, str message):
    # Generate the new name of the file
    cdef str s = os.sep
    cdef str new_name = 'data' + s + folder + s + tic + '-' + cik + '-' + filename

    # Close the file so that we can move it
    fh.close()
    os.rename(fn, new_name)
    print(message)


regex = re.compile(r"(.*\.sgml\s+?:\s+?|.*\nFILED AS OF DATE:\s+?)([\d]+?)\n.*", re.S)

# This function handles the cleaning of the 10-K
def clean(str fn):
    cdef bool error = False
    
    cdef str s = os.sep
    cdef str tic = fn.split(s)[1]
    cdef str cik = fn.split(s)[2]
    cdef str filename = fn.split(s)[4]
    cdef str report_key = "report:" + cik + ":" + fn
    cdef str cleaned_key = "cleaned:" + cik + ":" + fn

    # Open the file, get all of the content, and then pull it into a parser
    fh = open(fn, 'r')
    cdef unicode contents = fh.read()

    # Clean up some of the text to fix malformed HTML before parsing it
    cdef list malformed_tags = ['ACCEPTANCE-DATETIME', 'TYPE', 'SEQUENCE', 'FILENAME', 'DESCRIPTION']
    cdef str tag
    for tag in malformed_tags:
        # Do a regex that replaces all of these malformed tags in the document
        regex = re.compile(r"(\n<%s>[^<]*?)\n" % re.escape(tag), re.I)
        contents = regex.sub(r"\1</%s>\n" % tag, contents)

    # Create the parser
    parser = etree.HTMLParser()
    document = etree.parse(StringIO(contents), parser)
    doc = document.getroot()
    
    # The document can either have a root node of sec-document or ims-document
    if doc.xpath('//sec-document') is not None:
        root = doc.xpath('//sec-document[1]')[0]
    elif doc.xpath('//ims-document') is not None: 
        root = doc.xpath('//ims-document[1]')[0]
    elif doc.xpath('//document') is not None:
        root = doc.xpath('//document[1]')[0]
    elif doc.xpath('//error') is not None:
        root = None
    else:
        root = None
        
    if root is None:
        # Root node error 
        move_file(fh, fn, "_error", tic, cik, filename, "No root or erroneous root node - moved file")
        error = True
    if error: return error

    # Check if this is an amended 10-K and throw it out if so
    type_text = root.xpath('//type/text()')
    if type_text is None or len(type_text) == 0:
        move_file(fh, fn, "_error", tic, cik, filename, "Error finding type - moved file")
        error = True
    elif type_text[0] == '10-K/A':
        move_file(fh, fn, "_amended", tic, cik, filename, "Amended 10-K - moved file")
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
            filing_dt_text = re.sub(regex, r"\2", header_text)
        else:
            move_file(fh, fn, "_error", tic, cik, filename, "Bad filing date - moved file")
            error = True
        if error: return error
    else:
        # Get the filing date
        filing_dt_text = acc_dt[0].split('\n', 1)[0][:8]

    filing_dt = dt.strptime(filing_dt_text, '%Y%m%d')
    filing_ts = time.mktime(filing_dt.timetuple())
    begin_dt = dt(1995, 1, 1)

    # If the filing date is not within our date range, then move it
    if begin_dt > filing_dt:
        move_file(fh, fn, "_outofrange", tic, cik, filename, "Out of date range - moved file.")
        error = True
    if error: return error

    # See if we can find stock info for this company on the filing date of the 10-K
    cdef int index = 0
    cik_df = None
    try:
        index = df.index.get_loc((bytes(cik, 'utf-8'), filing_dt))
        cik_df = df.ix[bytes(cik, 'utf-8')]
        price = cik_df.ix[filing_dt, 'PRC']
        # Now, check if the price of the stock is less than $3.00
        if price < 3.0:
            move_file(fh, fn, "_nostockdata", tic, cik, filename, "Price less than $3.00 - moved file.")
            error = True
    except (IndexError, KeyError):
        # We couldn't find the cik or date for this 10-k
        move_file(fh, fn, "_nostockdata", tic, cik, filename, "No stock data found - moved file.")
        error = True
    if error: return error
    
    # Grab the report
    cdef str report = ''.join(root.xpath('//document/text')[0].itertext())

    # We will tokenize the text and iterate through each word
    cdef list tokens = report.split()
    cdef list keep_tokens = []
    cdef set stopwords_set = set(stopwords.words('english'))
    punc_table = str.maketrans("", "", string.punctuation)
    
    # Filter out words
    cdef str word
    for word in tokens:
        # Quick check to make sure we should keep filtering the word
        if len(word) != 1:
            # Strip punctuation from the word first and make it lowercase
            word = word.translate(punc_table).lower()

            # Add the word to the keep pile if it is not a stopword and if it is in 2of12inf dictionary
            if word not in stopwords_set and word in dict_2of12inf:
                keep_tokens.append(word)
            
    tokens = keep_tokens
    report = " ".join(tokens)
    cdef int total_words = len(tokens)

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
    rds.hmset(report_key, report_hash)
    rds.set(cleaned_key, time.time())


def process(str fn):
    
    cdef str s = os.sep
    cdef str tic = fn.split(s)[1]
    cdef str cik = fn.split(s)[2]
    cdef str report_key = "report:" + cik + ":" + fn
    cdef str processed_key = "processed:"+ cik + ":" + fn
    
    # Get the report out of redis
    #print("Found in redis: " + report_key)
    cdef str report = str(rds.hget(report_key, 'report'))
    filing_dt = dt.fromtimestamp(int(float(rds.hget(report_key, 'filing_date').decode('utf-8'))))
    cik_df = pickle.loads(rds.hget(report_key, 'company_data'))
    cdef int index = int(rds.hget(report_key, 'index'))
    cdef dict report_hash = {}

    # Now that everything is cleaned up, we can run the word processing algorithm
    pos_occurs = defaultdict(int)
    neg_occurs = defaultdict(int)
    negators = pd.Series(['not', 'no', 'never'])

    # We will tokenize the text and iterate through each word
    tokens = pd.Series(report.split())

    # Now, process the text
    cdef int i
    cdef str token, root, word
    cdef bool negated
    for i, token in tokens.iteritems():
        if token in pos_dict:
            # Check to see if there is a negator
            negated = False
            for word in tokens.iloc[(i - 3):(i + 3)]:
                if word in negators.values:
                    #print("Found a negator: " + word + " - " + token)
                    negated = True
            if not negated:
                root = pos_roots_map[token]
                pos_occurs[root] += 1
        elif token in neg_dict:
            # Check to see if there is a negator
            negated = False
            for word in tokens.iloc[(i - 3):(i + 3)]:
                if word in negators.values:
                    #print("Found a negator: " + word + " - " + token)
                    negated = True
            if not negated:
                root = neg_roots_map[token]
                neg_occurs[root] += 1

    # For the roots we didn't find, set frequency to zero
    for root in pos_roots:
        if root not in pos_occurs:
            pos_occurs[root] = 0
    for root in neg_roots:
        if root not in neg_occurs:
            neg_occurs[root] = 0
            
    # Use the index we found earlier to grab the historical info
    hist_returns = cik_df.ix[(index + 1):, 'RET']

    # Calculate the historical return before the filing date
    cdef float hist_ret = 1.0
    for col, series in hist_returns.iteritems():
        if col == 'RET':
            for r in series:
                if not math.isnan(r):
                    hist_ret *= (r + 1.0)
    hist_ret = hist_ret - 1.0
    #print("Historical return: " + str(hist_ret))

    # Use the index we found earlier to grab the four day window returns
    returns = cik_df.ix[(index - 3):(index + 1), ['RET','vwretd']]

    # Calculate the abnormal return: r_i = M{t=0, 3} (ret_i,j) - M{t=0,3} (ret_vwi,t)
    cdef float ret = 1.0
    cdef float ret_vwi = 1.0
    for col, series in returns.iteritems():
        if col == 'RET':
            for r in series:
                if not math.isnan(r):
                    ret *= (r + 1.0)
        elif col == 'vwretd':
            for r in series:
                if  not math.isnan(r):
                    ret_vwi *= (r + 1.0)
    cdef float ab_ret = ((ret - 1.0) - (ret_vwi - 1.0))
    #print("Abnormal return: " + str(ab_ret))

    # Save results of text processing to key in redis
    report_hash['pos_occurs'] = pickle.dumps(pos_occurs)
    report_hash['neg_occurs'] = pickle.dumps(neg_occurs)
    report_hash['hist_ret'] = hist_ret
    report_hash['ab_ret'] = ab_ret
    report_hash['mtime'] = time.time()

    print("Saving to redis: " + report_key)
    rds.hmset(report_key, report_hash)
    rds.set(processed_key, time.time())


def test():
    # This is for testing
    cdef int count = 1
    cdef int stop = 100000
    cdef bool skip_cleaned = True
    cdef bool skip_processed = True
    cdef bool process_file = True

    cdef dict report_hash
    cdef str fn, s, tic, cik, cleaned_key, processed_key, report_key
    cdef bool cleaned, processed, error

    cdef str dirpath
    cdef list dirnames, filenames
    cdef str folder = "SEC-Edgar-data"
    for (dirpath, dirnames, filenames) in os.walk(folder, topdown=False):
        for filename in filenames:
            report_hash = {}
            fn = os.sep.join([dirpath, filename])
            
            if filename.endswith('.txt'):# and filename == "0000950116-97-000637.txt":
                if count > stop:
                    break
                print(fn)
                s = os.sep
                tic = fn.split(s)[1]
                cik = fn.split(s)[2]
                
                # Check redis to see if we have processed or cleaned the report already
                cleaned_key = "cleaned:" + cik + ":" + fn
                processed_key = "processed:" + cik + ":" + fn
                report_key = "report:" + cik + ":" + fn
                (cleaned, processed) = check_redis(cleaned_key, processed_key, report_key)
                
                # If the report has been cleaned or we don't want to clean it anyway, skip this step
                error = False
                if not cleaned or not skip_cleaned:
                    print("(" + str(count) + ") Cleaning " + fn)
                    error = clean(fn)
                    
                    if not process and not error:
                        count += 1
                        continue
                if error: continue
                
                # After possibly cleaning, check if we should process the file
                if (not processed or not skip_processed) and process_file:
                    print("(" + str(count) + ") Processing " + fn)
                    process(fn)
                    
                    count += 1

def generate_yearly_data():
    count = 0
    stop = math.inf
    global yearly_data

    # Check if redis already has the yearly_data
    if rds.exists("yearly-data"):
        print("Found yearly data in Redis")
        yearly_data = pickle.loads(zlib.decompress(rds.get("yearly-data")))
        return

    keys = rds.keys("report:*")
    errors = []
    for key in keys:
        
        if count >= stop:
            break
            
        report_hash = rds.hgetall(key)
        try:
            year = 1
            cik = str(report_hash[b'cik'].decode('utf-8'))
            fn = str(report_hash[b'path'].decode('utf-8'))
            key = "report:" + cik + ":" + fn
            pos_occurs = pickle.loads(report_hash[b'pos_occurs'])
            neg_occurs = pickle.loads(report_hash[b'neg_occurs'])
            year = int(report_hash[b'year'])
            total_words = int(report_hash[b'total_words'])
            hist_ret = float(report_hash[b'hist_ret'])
            ab_ret = float(report_hash[b'ab_ret'])
            
            if total_words == 0:
                # Not sure why this would be zero, but we may need to reprocess
                print("Error with: " + key)
                cleaned_key = "cleaned:" + cik + ":" + fn
                processed_key = "processed:" + cik + ":" + fn
                
                # Delete this error from redis
                rds.delete(cleaned_key, processed_key, key)
                
                errors.append(key)
                continue
            
            try: yearly_data[year]
            except KeyError:
                yearly_data[year] = []
                
            year_list = yearly_data[year]
            year_list.append({
                'pos_occurs': pos_occurs,
                'neg_occurs': neg_occurs,
                'total_words': total_words,
                'hist_ret': hist_ret,
                'ab_ret': ab_ret
            })
            yearly_data[year] = year_list
            
            count += 1
        except KeyError:
            continue
        except e:
            print(e)
            
    if len(errors) > 0:
        print("Total errors: " + str(len(errors)))

    for year in sorted(yearly_data.keys()):
        print(year, len(yearly_data[year]))

    # Save yearly_data in redis
    print("Saving yearly data to Redis")
    rds.set("yearly-data", zlib.compress(pickle.dumps(yearly_data)))

def regression_analysis():
    start = 1997
    end = 2008
    skip_regressions = True
    skip_building = True
    global yearly_data

    # Generate a rolling training model using data up until year T-1
    for t in range(start, (end + 1)):
        
        print("Analyzing year " + str(t))

        if rds.exists("regression:" + str(t)) and skip_regressions:
            continue
        
        global pos_word_weights, neg_word_weights, hist_returns, ab_returns
        pos_word_weights = pd.DataFrame()
        neg_word_weights = pd.DataFrame()
        hist_returns = pd.DataFrame()
        ab_returns = pd.DataFrame()
        
        # Iterate over each year before year T and build the training data set
        for year in range((start - 1), t):

            key = "regression-data:" + str(year)

            if rds.exists(key) and skip_building:
                # Extract the data
                pos_word_weights = pickle.loads(zlib.decompress(rds.hget(key, 'pos_word_weights')))
                neg_word_weights = pickle.loads(zlib.decompress(rds.hget(key, 'neg_word_weights')))
                hist_returns = pickle.loads(zlib.decompress(rds.hget(key, 'hist_returns')))
                ab_returns = pickle.loads(zlib.decompress(rds.hget(key, 'ab_returns')))
                continue

            print("Building year " + str(year) + "/" + str(t))
            
            try: yearly_data[year]
            except KeyError:
                print("Year " + str(year) + " not found.")
                continue
            
            # Parallel process the initial word weights
            p = Pool()
            global count
            count = 0
            total = len(yearly_data[year]) 
            results = p.starmap(generate_weights, zip(yearly_data[year], range(1, total + 1), repeat(total)))
            results_df = pd.DataFrame.from_records(results, columns=['pos_weights', 'neg_weights', 'hist_ret', 'ab_ret'])

            for i in range(0, total):
                print("Count: " + str(i) + "/" + str(total))
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
            rds.hmset(key, reg_data_hash)
            
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
            rds.hmset(key, reg_hash)

# Iterate through each 10-K info for the year and generate the dataframe for the regression
def generate_weights(report, count, total):
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

if __name__ == '__main__':
    print("Generating yearly_data list")
    generate_yearly_data()

    print("Running regression analysis")
    regression_analysis()
