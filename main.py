from WordPower import WordPower

#start = 1995
start = 2008
end = 2008

if __name__ == '__main__':
    wp = WordPower(start, end)

    print("Loading data...")
    wp.load_data()

    print("Scraping SEC Edgar website for 10-Ks...")
    wp.scrape_edgar()

