import pandas as pd
import numpy as np
import re

from pandas.core.indexes import multi

PRODUCT_CATEGORIES = [
    'BOUCHERIE/TRAITEUR', 
    'ENTRETIEN/BAZAR', 
    'EPICERIE /BOISSONS', 
    'FRUITS/LEGUMES',
] 

'\d+'

NUMBERS = list(map(str, range(10)))

class MonoprixParser:
    
    def __init__(self):
        pass

    def rm_trailing_dots(self, line):
        pattern = '[\s,.][,.]|[,.][\s,.]'
        if re.search(pattern, line) is None:
            return False, line
        return True, re.split(pattern, line)


    def parse_shop_name(self, lines):
        self.shop_names = {'brand': lines[0], 'name': lines[1]} 

    def parse_adress(self, lines):
        self.adress = {
            'adress': lines[0], 
            'zip': lines[1], 
            'phone': lines[2]} 

    def parse_greetings(self, lines):
        self.greetings = lines[0]

    def get_category(self, line):
        dots, line = self.rm_trailing_dots(line)
        if dots:
            print('Trailing dot detected')
        for category in PRODUCT_CATEGORIES:
            if category in line:
                return category
        return None

    def get_multiplicity(self, line):
        if line[0] not in  NUMBERS:
            return 1, line
        return int(line[0]), line[4:]

    # Not including € for robustness, but might have false positive then ?
    def get_prices(self, line):
        pattern = '\d+[,.]\d{2}'
        prices = [float(price.replace(',', '.')) for price in re.findall(pattern, line)]
        line = re.split(pattern, line)[0]
        return prices, line
        

    def parse_articles(self, lines):
        columns = [
            'category', 'name', 'multiplicity', 'unit_price', 'dis_price', 'disc'
        ]

        cur_category = np.nan
        df_lines = []
        for line in lines:
            print('Processing line:', line)

            category = self.get_category(line)
            if category is not None:
                cur_category = category
                continue

            multiplicity, line = self.get_multiplicity(line)

            prices, line = self.get_prices(line)
            if not prices:
                print('No Price !')
                unit_price = np.nan
            else:
                unit_price = prices[0]
                if len(prices) == 2 :
                    if np.isnan(multiplicity):
                        print('2 prices but no multiplicity')
                    mul_price = unit_price * multiplicity
                    if mul_price != prices[1]:
                        print('Multiplicity price and second price !=')
                        print('unit_price: ', unit_price)
                        print('second_price: ', prices[1])
                        print('multiplicity: ', multiplicity)
                elif len(prices) > 2:
                    print('More than 2 prices')

            article_name = line
            disc = np.nan
            disc_price = np.nan
            df_lines.append([cur_category, article_name, multiplicity, unit_price, disc_price, disc])
            print()
        self.df_articles = pd.DataFrame(df_lines, columns=columns)
        return 


    def parse_monoprix(self, lines):
        self.parse_shop_name(lines[:2])
        self.parse_adress(lines[2:5])
        self.parse_greetings(lines[5:6])
        self.parse_articles(lines[6:])