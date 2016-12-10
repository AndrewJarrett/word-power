from WordPower import WordPower

#start = 1995
start = 2008
end = 2008

scrape = True
process = True
regression = True
reports = True

if __name__ == '__main__':
    wp = WordPower(start, end)

    print("Running algorithm for " + str(start) + " - " + str(end) + "\n")

    print("Loading data...")
    wp.load_data()

    if scrape:
        print("\nScraping SEC Edgar website for 10-Ks...")
        wp.scrape_edgar()

    if process:
        print("\nProcessing the 10-K files...")
        wp.process_files()

    if regression:
        print("\nRunning the regression analysis...")
        wp.regression_analysis()

    if reports:
        print("\nRunning the reports...")
        wp.run_reports()

