
from __future__ import absolute_import
#typing
from datetime import datetime
import re

from collections import defaultdict
from allennlp.data.tokenizers import Token
try:
    from itertools import izip
except:
    izip = zip


TWELVE_TO_TWENTY_FOUR = 1200
HOUR_TO_TWENTY_FOUR = 100
HOURS_IN_DAY = 2400
AROUND_RANGE = 30

APPROX_WORDS = [u'about', u'around', u'approximately']
WORDS_PRECEDING_TIME = [u'at', u'between', u'to', u'before', u'after']

def get_times_from_utterance(utterance     ,
                             char_offset_to_token_index                ,
                             indices_of_approximate_words          )                        :
    u"""
    Given an utterance, we get the numbers that correspond to times and convert them to
    values that may appear in the query. For example: convert ``7pm`` to ``1900``.
    """

    pm_linking_dict = _time_regex_match(ur'\d+pm',
                                        utterance,
                                        char_offset_to_token_index,
                                        lambda match: [int(match.rstrip(u'pm'))
                                                       * HOUR_TO_TWENTY_FOUR +
                                                       TWELVE_TO_TWENTY_FOUR],
                                        indices_of_approximate_words)

    am_linking_dict = _time_regex_match(ur'\d+am',
                                        utterance,
                                        char_offset_to_token_index,
                                        lambda match: [int(match.rstrip(u'am'))
                                                       * HOUR_TO_TWENTY_FOUR],
                                        indices_of_approximate_words)

    oclock_linking_dict = _time_regex_match(ur"\d+\so'clock",
                                            utterance,
                                            char_offset_to_token_index,
                                            lambda match: digit_to_query_time(match.rstrip(u"o'clock")),
                                            indices_of_approximate_words)

    times_linking_dict                       = defaultdict(list)
    linking_dicts = [pm_linking_dict, am_linking_dict, oclock_linking_dict]

    for linking_dict in linking_dicts:
        for key, value in list(linking_dict.items()):
            times_linking_dict[key].extend(value)

    return times_linking_dict

def get_date_from_utterance(tokenized_utterance             ,
                            year      = 1993,
                            month      = None,
                            day      = None)            :
    u"""
    When the year is not explicitly mentioned in the utterance, the query assumes that
    it is 1993 so we do the same here. If there is no mention of the month or day then
    we do not return any dates from the utterance.
    """
    utterance = u' '.join([token.text for token in tokenized_utterance])
    year_result = re.findall(ur'199[0-4]', utterance)
    if year_result:
        year = int(year_result[0])

    for token in tokenized_utterance:
        if token.text in MONTH_NUMBERS:
            month = MONTH_NUMBERS[token.text]
        if token.text in DAY_NUMBERS:
            day = DAY_NUMBERS[token.text]

    for tens, digits in izip(tokenized_utterance, tokenized_utterance[1:]):
        bigram = u' '.join([tens.text, digits.text])
        if bigram in DAY_NUMBERS:
            day = DAY_NUMBERS[bigram]
    if month and day:
        return datetime(year, month, day)
    return None

def get_numbers_from_utterance(utterance     , tokenized_utterance             )                        :
    u"""
    Given an utterance, this function finds all the numbers that are in the action space. Since we need to
    keep track of linking scores, we represent the numbers as a dictionary, where the keys are the string
    representation of the number and the values are lists of the token indices that triggers that number.
    """
    # When we use a regex to find numbers or strings, we need a mapping from
    # the character to which token triggered it.
    char_offset_to_token_index = dict((token.idx, token_index)
                                  for token_index, token in enumerate(tokenized_utterance))

    # We want to look up later for each time whether it appears after a word
    # such as "about" or "approximately".
    indices_of_approximate_words = set(index for index, token in enumerate(tokenized_utterance)
                                    if token.text in APPROX_WORDS)

    indices_of_words_preceding_time = set(index for index, token in enumerate(tokenized_utterance)
                                       if token.text in WORDS_PRECEDING_TIME)

    number_linking_dict                       = defaultdict(list)
    for token_index, token in enumerate(tokenized_utterance):
        if token.text.isdigit():
            if token_index - 1 in indices_of_words_preceding_time:
                for time in digit_to_query_time(token.text):
                    number_linking_dict[unicode(time)].append(token_index)
            else:
                number_linking_dict[token.text].append(token_index)

    times_linking_dict = get_times_from_utterance(utterance,
                                                  char_offset_to_token_index,
                                                  indices_of_approximate_words)

    for key, value in list(times_linking_dict.items()):
        number_linking_dict[key].extend(value)

    for index, token in enumerate(tokenized_utterance):
        for number in NUMBER_TRIGGER_DICT.get(token.text, []):
            number_linking_dict[number].append(index)

    for tens, digits in izip(tokenized_utterance, tokenized_utterance[1:]):
        bigram = u' '.join([tens.text, digits.text])
        if bigram in DAY_NUMBERS:
            number_linking_dict[unicode(DAY_NUMBERS[bigram])].append(len(tokenized_utterance) - 1)

    return number_linking_dict

def digit_to_query_time(digit     )             :
    u"""
    Given a digit in the utterance, return a list of the times that it corresponds to.
    """
    if int(digit) % 12 == 0:
        return [0, 1200, 2400]
    return [int(digit) * HOUR_TO_TWENTY_FOUR,
            (int(digit) * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR) % HOURS_IN_DAY]

def get_approximate_times(times           )             :
    u"""
    Given a list of times that follow a word such as ``about``,
    we return a list of times that could appear in the query as a result
    of this. For example if ``about 7pm`` appears in the utterance, then
    we also want to add ``1830`` and ``1930``.
    """
    approximate_times = []
    for time in times:
        approximate_times.append((time + AROUND_RANGE) % HOURS_IN_DAY)
        # The number system is not base 10 here, there are 60 minutes
        # in an hour, so we can't simply add time - AROUND_RANGE.
        approximate_times.append((time - HOUR_TO_TWENTY_FOUR + AROUND_RANGE) % HOURS_IN_DAY)
    return approximate_times

def _time_regex_match(regex     ,
                      utterance     ,
                      char_offset_to_token_index                ,
                      map_match_to_query_value                            ,
                      indices_of_approximate_words          )                        :
    ur"""
    Given a regex for matching times in the utterance, we want to convert the matches
    to the values that appear in the query and token indices they correspond to.

    ``char_offset_to_token_index`` is a dictionary that maps from the character offset to
    the token index, we use this to look up what token a regex match corresponds to.
    ``indices_of_approximate_words`` are the token indices of the words such as ``about`` or
    ``approximately``. We use this to check if a regex match is preceded by one of these words.
    If it is, we also want to add the times that define this approximate time range.

    ``map_match_to_query_value`` is a function that converts the regex matches to the
    values that appear in the query. For example, we may pass in a regex such as ``\d+pm``
    that matches times such as ``7pm``. ``map_match_to_query_value`` would be a function that
    takes ``7pm`` as input and returns ``1900``.
    """
    linking_scores_dict                       = defaultdict(list)
    number_regex = re.compile(regex)
    for match in number_regex.finditer(utterance):
        query_values = map_match_to_query_value(match.group())
        # If the time appears after a word like ``about`` then we also add
        # the times that mark the start and end of the allowed range.
        approximate_times = []
        if char_offset_to_token_index.get(match.start(), 0) - 1 in indices_of_approximate_words:
            approximate_times.extend(get_approximate_times(query_values))
        query_values.extend(approximate_times)
        if match.start() in char_offset_to_token_index:
            for query_value in query_values:
                linking_scores_dict[unicode(query_value)].append(char_offset_to_token_index[match.start()])
    return linking_scores_dict

def get_trigger_dict(trigger_lists                 ,
                     trigger_dicts                            )                        :
    merged_trigger_dict                       = defaultdict(list)
    for trigger_list in trigger_lists:
        for trigger in trigger_list:
            merged_trigger_dict[trigger.lower()].append(trigger)

    for trigger_dict in trigger_dicts:
        for key, value in list(trigger_dict.items()):
            merged_trigger_dict[key.lower()].extend(value)

    return merged_trigger_dict

def convert_to_string_list_value_dict(trigger_dict                )                        :
    return dict((key, [unicode(value)]) for key, value in list(trigger_dict.items()))

AIRLINE_CODES = {u'alaska': [u'AS'],
                 u'alliance': [u'3J'],
                 u'alpha': [u'7V'],
                 u'america west': [u'HP'],
                 u'american': [u'AA'],
                 u'american trans': [u'TZ'],
                 u'argentina': [u'AR'],
                 u'atlantic': [u'DH'],
                 u'atlantic.': [u'EV'],
                 u'braniff.': [u'BE'],
                 u'british': [u'BA'],
                 u'business': [u'HQ'],
                 u'canada': [u'AC'],
                 u'canadian': [u'CP'],
                 u'carnival': [u'KW'],
                 u'christman': [u'SX'],
                 u'colgan': [u'9L'],
                 u'comair': [u'OH'],
                 u'continental': [u'CO'],
                 u'czecho': [u'OK'],
                 u'delta': [u'DL'],
                 u'eastern': [u'EA'],
                 u'express': [u'9E'],
                 u'grand': [u'QD'],
                 u'lufthansa': [u'LH'],
                 u'mesaba': [u'XJ'],
                 u'mgm': [u'MG'],
                 u'midwest': [u'YX'],
                 u'nation': [u'NX'],
                 u'northeast': [u'2V'],
                 u'northwest': [u'NW'],
                 u'ontario': [u'GX'],
                 u'ontario express': [u'9X'],
                 u'precision': [u'RP'],
                 u'royal': [u'AT'],
                 u'sabena': [u'SN'],
                 u'sky': [u'OO'],
                 u'south': [u'WN'],
                 u'states': [u'9N'],
                 u'thai': [u'TG'],
                 u'tower': [u'FF'],
                 u'twa': [u'TW'],
                 u'united': [u'UA'],
                 u'us': [u'US'],
                 u'west': [u'OE'],
                 u'wisconson': [u'ZW'],
                 u'world': [u'RZ']}

CITY_CODES = {u'ATLANTA': [u'MATL'],
              u'BALTIMORE': [u'BBWI'],
              u'BOSTON': [u'BBOS'],
              u'BURBANK': [u'BBUR'],
              u'CHARLOTTE': [u'CCLT'],
              u'CHICAGO': [u'CCHI'],
              u'CINCINNATI': [u'CCVG'],
              u'CLEVELAND': [u'CCLE'],
              u'COLUMBUS': [u'CCMH'],
              u'DALLAS': [u'DDFW'],
              u'DENVER': [u'DDEN'],
              u'DETROIT': [u'DDTT'],
              u'FORT WORTH': [u'FDFW'],
              u'HOUSTON': [u'HHOU'],
              u'KANSAS CITY': [u'MMKC'],
              u'LAS VEGAS': [u'LLAS'],
              u'LONG BEACH': [u'LLGB'],
              u'LOS ANGELES': [u'LLAX'],
              u'MEMPHIS': [u'MMEM'],
              u'MIAMI': [u'MMIA'],
              u'MILWAUKEE': [u'MMKE'],
              u'MINNEAPOLIS': [u'MMSP'],
              u'MONTREAL': [u'YYMQ'],
              u'NASHVILLE': [u'BBNA'],
              u'NEW YORK': [u'NNYC'],
              u'NEWARK': [u'JNYC'],
              u'OAKLAND': [u'OOAK'],
              u'ONTARIO': [u'OONT'],
              u'ORLANDO': [u'OORL'],
              u'PHILADELPHIA': [u'PPHL'],
              u'PHOENIX': [u'PPHX'],
              u'PITTSBURGH': [u'PPIT'],
              u'SALT LAKE CITY': [u'SSLC'],
              u'SAN DIEGO': [u'SSAN'],
              u'SAN FRANCISCO': [u'SSFO'],
              u'SAN JOSE': [u'SSJC'],
              u'SEATTLE': [u'SSEA'],
              u'ST. LOUIS': [u'SSTL'],
              u'ST. PAUL': [u'SMSP'],
              u'ST. PETERSBURG': [u'STPA'],
              u'TACOMA': [u'TSEA'],
              u'TAMPA': [u'TTPA'],
              u'TORONTO': [u'YYTO'],
              u'WASHINGTON': [u'WWAS'],
              u'WESTCHESTER COUNTY': [u'HHPN']}

MONTH_NUMBERS = {u'january': 1,
                 u'february': 2,
                 u'march': 3,
                 u'april': 4,
                 u'may': 5,
                 u'june': 6,
                 u'july': 7,
                 u'august': 8,
                 u'september': 9,
                 u'october': 10,
                 u'november': 11,
                 u'december': 12}

DAY_NUMBERS = {u'first': 1,
               u'second': 2,
               u'third': 3,
               u'fourth': 4,
               u'fifth': 5,
               u'sixth': 6,
               u'seventh': 7,
               u'eighth': 8,
               u'ninth': 9,
               u'tenth': 10,
               u'eleventh': 11,
               u'twelfth': 12,
               u'thirteenth': 13,
               u'fourteenth': 14,
               u'fifteenth': 15,
               u'sixteenth': 16,
               u'seventeenth': 17,
               u'eighteenth': 18,
               u'nineteenth': 19,
               u'twentieth': 20,
               u'twenty first': 21,
               u'twenty second': 22,
               u'twenty third': 23,
               u'twenty fourth': 24,
               u'twenty fifth': 25,
               u'twenty sixth': 26,
               u'twenty seventh': 27,
               u'twenty eighth': 28,
               u'twenty ninth': 29,
               u'thirtieth': 30,
               u'thirty first': 31}

GROUND_SERVICE = {u'air taxi': [u'AIR TAXI OPERATION'],
                  u'car': [u'RENTAL CAR'],
                  u'limo': [u'LIMOUSINE'],
                  u'rapid': [u'RAPID TRANSIT'],
                  u'rental': [u'RENTAL CAR'],
                  u'taxi': [u'TAXI']}

MISC_STR = {u"every day" : [u"DAILY"]}

MISC_TIME_TRIGGERS = {u'morning': [u'0', u'1200'],
                      u'afternoon': [u'1200', u'1800'],
                      u'after': [u'1200', u'1800'],
                      u'evening': [u'1800', u'2200'],
                      u'late evening': [u'2000', u'2200'],
                      u'lunch': [u'1400'],
                      u'noon': [u'1200']}

TABLES = {u'aircraft': [u'aircraft_code', u'aircraft_description',
                       u'manufacturer', u'basic_type', u'propulsion',
                       u'wide_body', u'pressurized'],
          u'airline': [u'airline_name', u'airline_code'],
          u'airport': [u'airport_code', u'airport_name', u'airport_location',
                      u'state_code', u'country_name', u'time_zone_code',
                      u'minimum_connect_time'],
          u'airport_service': [u'city_code', u'airport_code', u'miles_distant',
                              u'direction', u'minutes_distant'],
          u'city': [u'city_code', u'city_name', u'state_code', u'country_name', u'time_zone_code'],
          u'class_of_service': [u'booking_class', u'rank', u'class_description'],
          u'date_day': [u'month_number', u'day_number', u'year', u'day_name'],
          u'days': [u'days_code', u'day_name'],
          u'equipment_sequence': [u'aircraft_code_sequence', u'aircraft_code'],
          u'fare': [u'fare_id', u'from_airport', u'to_airport', u'fare_basis_code',
                   u'fare_airline', u'restriction_code', u'one_direction_cost',
                   u'round_trip_cost', u'round_trip_required'],
          u'fare_basis': [u'fare_basis_code', u'booking_class', u'class_type', u'premium', u'economy',
                         u'discounted', u'night', u'season', u'basis_days'],
          u'flight': [u'flight_id', u'flight_days', u'from_airport', u'to_airport', u'departure_time',
                     u'arrival_time', u'airline_flight', u'airline_code', u'flight_number',
                     u'aircraft_code_sequence', u'meal_code', u'stops', u'connections',
                     u'dual_carrier', u'time_elapsed'],
          u'flight_fare': [u'flight_id', u'fare_id'],
          u'flight_leg': [u'flight_id', u'leg_number', u'leg_flight'],
          u'flight_stop': [u'flight_id', u'stop_number', u'stop_days', u'stop_airport',
                          u'arrival_time', u'arrival_airline', u'arrival_flight_number',
                          u'departure_time', u'departure_airline', u'departure_flight_number',
                          u'stop_time'],
          u'food_service': [u'meal_code', u'meal_number', u'compartment', u'meal_description'],
          u'ground_service': [u'city_code', u'airport_code', u'transport_type', u'ground_fare'],
          u'month': [u'month_number', u'month_name'],
          u'restriction': [u'restriction_code', u'advance_purchase', u'stopovers',
                          u'saturday_stay_required', u'minimum_stay', u'maximum_stay',
                          u'application', u'no_discounts'],
          u'state': [u'state_code', u'state_name', u'country_name']}

DAY_OF_WEEK_DICT = {u'weekdays' : [u'MONDAY', u'TUESDAY', u'WEDNESDAY', u'THURSDAY', u'FRIDAY']}

YES_NO = {u'one way': [u'NO'],
          u'economy': [u'YES']}

CITY_AIRPORT_CODES = {u'atlanta' : [u'ATL'],
                      u'boston' : [u'BOS'],
                      u'baltimore': [u'BWI'],
                      u'charlotte': [u'CLT'],
                      u'dallas': [u'DFW'],
                      u'detroit': [u'DTW'],
                      u'la guardia': [u'LGA'],
                      u'oakland': [u'OAK'],
                      u'philadelphia': [u'PHL'],
                      u'pittsburgh': [u'PIT'],
                      u'san francisco': [u'SFO'],
                      u'toronto': [u'YYZ']}

AIRPORT_CODES = [u'ATL', u'NA', u'OS', u'UR', u'WI', u'CLE', u'CLT', u'CMH',
                 u'CVG', u'DAL', u'DCA', u'DEN', u'DET', u'DFW', u'DTW',
                 u'EWR', u'HOU', u'HPN', u'IAD', u'IAH', u'IND', u'JFK',
                 u'LAS', u'LAX', u'LGA', u'LG', u'MCI', u'MCO', u'MDW', u'MEM',
                 u'MIA', u'MKE', u'MSP', u'OAK', u'ONT', u'ORD', u'PHL', u'PHX',
                 u'PIE', u'PIT', u'SAN', u'SEA', u'SFO', u'SJC', u'SLC',
                 u'STL', u'TPA', u'YKZ', u'YMX', u'YTZ', u'YUL', u'YYZ']

AIRLINE_CODE_LIST = [u'AR', u'3J', u'AC', u'9X', u'ZW', u'AS', u'7V',
                     u'AA', u'TZ', u'HP', u'DH', u'EV', u'BE', u'BA',
                     u'HQ', u'CP', u'KW', u'SX', u'9L', u'OH', u'CO',
                     u'OK', u'DL', u'9E', u'QD', u'LH', u'XJ', u'MG',
                     u'YX', u'NX', u'2V', u'NW', u'RP', u'AT', u'SN',
                     u'OO', u'WN', u'TG', u'FF', u'9N', u'TW', u'RZ',
                     u'UA', u'US', u'OE']

CITIES = [u'NASHVILLE', u'BOSTON', u'BURBANK', u'BALTIMORE', u'CHICAGO', u'CLEVELAND',
          u'CHARLOTTE', u'COLUMBUS', u'CINCINNATI', u'DENVER', u'DALLAS', u'DETROIT',
          u'FORT WORTH', u'HOUSTON', u'WESTCHESTER COUNTY', u'INDIANAPOLIS', u'NEWARK',
          u'LAS VEGAS', u'LOS ANGELES', u'LONG BEACH', u'ATLANTA', u'MEMPHIS', u'MIAMI',
          u'KANSAS CITY', u'MILWAUKEE', u'MINNEAPOLIS', u'NEW YORK', u'OAKLAND', u'ONTARIO',
          u'ORLANDO', u'PHILADELPHIA', u'PHOENIX', u'PITTSBURGH', u'ST. PAUL', u'SAN DIEGO',
          u'SEATTLE', u'SAN FRANCISCO', u'SAN JOSE', u'SALT LAKE CITY', u'ST. LOUIS',
          u'ST. PETERSBURG', u'TACOMA', u'TAMPA', u'WASHINGTON', u'MONTREAL', u'TORONTO']

CITY_CODE_LIST = [u'BBNA', u'BBOS', u'BBUR', u'BBWI', u'CCHI', u'CCLE', u'CCLT', u'CCMH', u'CCVG', u'DDEN',
                  u'DDFW', u'DDTT', u'FDFW', u'HHOU', u'HHPN', u'IIND', u'JNYC', u'LLAS', u'LLAX', u'LLGB',
                  u'MATL', u'MMEM', u'MMIA', u'MMKC', u'MMKE', u'MMSP', u'NNYC', u'OOAK', u'OONT', u'OORL',
                  u'PPHL', u'PPHX', u'PPIT', u'SMSP', u'SSAN', u'SSEA', u'SSFO', u'SSJC', u'SSLC', u'SSTL',
                  u'STPA', u'TSEA', u'TTPA', u'WWAS', u'YYMQ', u'YYTO']

CLASS = [u'COACH', u'BUSINESS', u'FIRST', u'THRIST', u'STANDARD', u'SHUTTLE']

DAY_OF_WEEK = [u'MONDAY', u'TUESDAY', u'WEDNESDAY', u'THURSDAY', u'FRIDAY', u'SATURDAY', u'SUNDAY']

FARE_BASIS_CODE = [u'B', u'BH', u'BHW', u'BHX', u'BL', u'BLW', u'BLX', u'BN', u'BOW', u'BOX',
                   u'BW', u'BX', u'C', u'CN', u'F', u'FN', u'H', u'HH', u'HHW', u'HHX', u'HL', u'HLW', u'HLX',
                   u'HOW', u'HOX', u'J', u'K', u'KH', u'KL', u'KN', u'LX', u'M', u'MH', u'ML', u'MOW', u'P',
                   u'Q', u'QH', u'QHW', u'QHX', u'QLW', u'QLX', u'QO', u'QOW', u'QOX', u'QW', u'QX', u'S',
                   u'U', u'V', u'VHW', u'VHX', u'VW', u'VX', u'Y', u'YH', u'YL', u'YN', u'YW', u'YX']

MEALS = [u'BREAKFAST', u'LUNCH', u'SNACK', u'DINNER']

RESTRICT_CODES = [u'AP/2', u'AP/6', u'AP/12', u'AP/20', u'AP/21', u'AP/57', u'AP/58', u'AP/60',
                  u'AP/75', u'EX/9', u'EX/13', u'EX/14', u'EX/17', u'EX/19']

STATES = [u'ARIZONA', u'CALIFORNIA', u'COLORADO', u'DISTRICT OF COLUMBIA',
          u'FLORIDA', u'GEORGIA', u'ILLINOIS', u'INDIANA', u'MASSACHUSETTS',
          u'MARYLAND', u'MICHIGAN', u'MINNESOTA', u'MISSOURI', u'NORTH CAROLINA',
          u'NEW JERSEY', u'NEVADA', u'NEW YORK', u'OHIO', u'ONTARIO', u'PENNSYLVANIA',
          u'QUEBEC', u'TENNESSEE', u'TEXAS', u'UTAH', u'WASHINGTON', u'WISCONSIN']

STATE_CODES = [u'TN', u'MA', u'CA', u'MD', u'IL', u'OH', u'NC', u'CO', u'TX', u'MI', u'NY',
               u'IN', u'NJ', u'NV', u'GA', u'FL', u'MO', u'WI', u'MN', u'PA', u'AZ', u'WA',
               u'UT', u'DC', u'PQ', u'ON']

DAY_OF_WEEK_INDEX = dict((idx, [day]) for idx, day in enumerate(DAY_OF_WEEK))

TRIGGER_LISTS = [CITIES, AIRPORT_CODES,
                 STATES, STATE_CODES,
                 FARE_BASIS_CODE, CLASS,
                 AIRLINE_CODE_LIST, DAY_OF_WEEK,
                 CITY_CODE_LIST, MEALS,
                 RESTRICT_CODES]

TRIGGER_DICTS = [CITY_AIRPORT_CODES,
                 AIRLINE_CODES,
                 CITY_CODES,
                 GROUND_SERVICE,
                 DAY_OF_WEEK_DICT,
                 YES_NO,
                 MISC_STR]

ATIS_TRIGGER_DICT = get_trigger_dict(TRIGGER_LISTS, TRIGGER_DICTS)

NUMBER_TRIGGER_DICT                       = get_trigger_dict([], [convert_to_string_list_value_dict(MONTH_NUMBERS),
                                                                  convert_to_string_list_value_dict(DAY_NUMBERS),
                                                                  MISC_TIME_TRIGGERS])
