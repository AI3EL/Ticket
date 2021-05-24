import pandas as pd
import numpy as np
import re

PRODUCT_CATEGORIES = [
    'BOUCHERIE/TRAITEUR', 
    'ENTRETIEN/BAZAR', 
    'EPICERIE/BOISSONS', 
    'FRUITS/LEGUMES',
    'SURGELES/PRODUITS FRAIS'
] 
NUMBERS = list(map(str, range(10)))


class MonoprixParser:
    
    def __init__(self, verbose=False):
        self.verbose = verbose

    def parse_monoprix(self, lines):
        self.parse_shop_name(lines[:2])
        self.parse_adress(lines[2:5])
        self.parse_greetings(lines[5:6])
        self.parse_articles(lines[6:])


    def parse_shop_name(self, lines):
        self.shop_names = {'brand': lines[0], 'name': lines[1]} 


    def parse_adress(self, lines):
        self.adress = {
            'adress': lines[0], 
            'zip': lines[1], 
            'phone': lines[2]} 


    def parse_greetings(self, lines):
        self.greetings = lines[0]


    def parse_articles(self, lines):

        columns = [
            'category', 'name', 'multiplicity', 'unit_price', 
            'disc_price', 'disc_percent', 'is_second_disc', 'disc'
        ]

        cur_category = np.nan
        df_articles = []
        for line in lines:
            self.vprint('Processing line:', line)
            df_row = {}

            category = self.get_category(line)
            if category is not None:
                cur_category = category
                continue
            df_row['category'] = cur_category

            discount_dict = self.parse_discount_line(line)
            if discount_dict is not None:
                discount_dict['category'] = cur_category
                df_articles.append(discount_dict)
                continue
            
            self.parse_article_line(line, df_row)
            df_articles.append(df_row)
            self.vprint()

        self.df_articles = pd.DataFrame(df_articles, columns=columns)


    def get_category(self, line):
        dots, line = self.rm_trailing_dots(line)
        if dots:
            self.vprint('Trailing dot detected')
        for category in PRODUCT_CATEGORIES:
            if category in line:
                return category
        return None

    # Remove the trailing dots in category lines
    def rm_trailing_dots(self, line):
        pattern = '[\s,.][,.]|[,.][\s,.]'
        if re.search(pattern, line) is None:
            return False, line
        return True, re.split(pattern, line)

    # Get the product multiplicity and remove it
    def get_multiplicity(self, line):
        if line[0] not in  NUMBERS:
            return 1, line
        return int(line[0]), line[4:]


    # Parse what should be a normal discount line
    def parse_discount_line(self, line):
        patterns = {
            'disc_price': '-\d+[,.]\d{2}',
            'disc_percent': '\d{1,2}%',
            'is_second_disc': '(2e|2eme)',
        }
        pattern_matches = {pat_name: re.findall(pattern, line) for pat_name, pattern in patterns.items()}

        if not any(pattern_matches.values()):
            return None

        for pat_name in patterns:
            if len(pattern_matches[pat_name]) > 1:
                self.vprint(f'Too many patterns detected for {pat_name} pattern:', pattern_matches[pat_name])

        if pattern_matches['disc_price']:
            pattern_matches['disc_price'] = float(pattern_matches['disc_price'][0][1:].replace(',', '.'))
        else:
            pattern_matches['disc_price'] = np.nan

        if pattern_matches['disc_percent']:
            pattern_matches['disc_percent']  = float(pattern_matches['disc_percent'][0][:-1])
        else:
            pattern_matches['disc_percent'] = np.nan

        pattern_matches['is_second_disc'] = len(pattern_matches['is_second_disc'])>0
        
        pattern_matches['disc'] = line
        
        return pattern_matches
    
    
    # Parse what should be a normal article line
    def parse_article_line(self, line, df_row):
        multiplicity, line = self.get_multiplicity(line)

        prices, line = self.get_prices(line)
        if not prices:
            self.vprint('No Price !')
            unit_price = np.nan
        else:
            unit_price = prices[0]
            if len(prices) == 2 :

                if np.isnan(multiplicity):
                    self.vprint('2 prices but no multiplicity')
                mul_price = unit_price * multiplicity

                if mul_price != prices[1]:
                    self.vprint('Multiplicity price and second price !=')
                    self.vprint('unit_price: ', unit_price)
                    self.vprint('second_price: ', prices[1])
                    self.vprint('multiplicity: ', multiplicity)

            elif len(prices) > 2:
                self.vprint('More than 2 prices')

        df_row['name'] = line
        df_row['multiplicity'] = multiplicity
        df_row['unit_price'] = unit_price
        return 


    # Not including € for robustness, but might have false positive then ?
    def get_prices(self, line):
        pattern = '\d+[,.]\d{2}'
        prices = [float(price.replace(',', '.')) for price in re.findall(pattern, line)]
        line = re.split(pattern, line)[0]
        return prices, line


    def vprint(self, *args):
        if self.verbose:
            print(*args)













