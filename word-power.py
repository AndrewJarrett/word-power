from multiprocessing import Pool
import redis
import math
import pickle
from itertools import repeat

import pandas as pd
from pandas import Series, DataFrame

import statsmodels.api as sm

rds = redis.Redis()
count = 0
yearly_data = {}

pos_word_weights = pd.DataFrame()
neg_word_weights = pd.DataFrame()
hist_returns = pd.DataFrame()
ab_returns = pd.DataFrame()

def generate_yearly_data():
    count = 0
    stop = math.inf
    global yearly_data

    # Check if redis already has the yearly_data
    if rds.exists("yearly-data"):
        print("Found yearly data in Redis")
        yearly_data = pickle.loads(rds.get("yearly-data"))
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
    rds.set("yearly-data", pickle.dumps(yearly_data))

def regression_analysis():
    start = 1997
    end = 2008
    skip_regressions = False
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
                pos_word_weights = pickle.loads(rds.hget(key, 'pos_word_weights'))
                neg_word_weights = pickle.loads(rds.hget(key, 'neg_word_weights'))
                hist_returns = pickle.loads(rds.hget(key, 'hist_returns'))
                ab_returns = pickle.loads(rds.hget(key, 'ab_returns'))
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

            for (pos_weights, neg_weights, hist_ret, ab_ret) in results:
                pos_word_weights = pos_word_weights.append(pos_weights, ignore_index=True)
                neg_word_weights = neg_word_weights.append(neg_weights, ignore_index=True)
                hist_returns = hist_returns.append({'hist_ret': hist_ret}, ignore_index=True)
                ab_returns = ab_returns.append({'ab_ret': ab_ret}, ignore_index=True)

            # Save our progress to redis
            print("Saving progress for year " + str(year) + " to redis.")
            reg_data_hash = {
                'pos_word_weights': pickle.dumps(pos_word_weights),
                'neg_word_weights': pickle.dumps(neg_word_weights),
                'hist_returns': pickle.dumps(hist_returns),
                'ab_returns': pickle.dumps(ab_returns)
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
                'pos_model': pickle.dumps(pos_model),
                'neg_model': pickle.dumps(neg_model),
                'pos_ab_model': pickle.dumps(pos_ab_model),
                'neg_ab_model': pickle.dumps(neg_ab_model),
                'pos_coeffs': pickle.dumps(pos_coeffs),
                'neg_coeffs': pickle.dumps(neg_coeffs)
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

    return (pos_weights, neg_weights, hist_ret, ab_ret)

if __name__ == '__main__':
    print("Generating yearly_data list")
    generate_yearly_data()

    print("Running regression analysis")
    regression_analysis()
