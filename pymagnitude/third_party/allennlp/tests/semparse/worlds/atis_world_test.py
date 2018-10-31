

from __future__ import division
from __future__ import absolute_import
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.atis_world import AtisWorld
from io import open

class TestAtisWorld(AllenNlpTestCase):
    def setUp(self):
        super(TestAtisWorld, self).setUp()
        test_filename = self.FIXTURES_ROOT / u"data" / u"atis" / u"sample.json"
        self.data = open(test_filename).readlines()

    def test_atis_global_actions(self): # pylint: disable=no-self-use
        world = AtisWorld([])
        valid_actions = world.valid_actions

        assert set(valid_actions.keys()) == set([u'agg',
                                             u'agg_func',
                                             u'agg_results',
                                             u'biexpr',
                                             u'binaryop',
                                             u'boolean',
                                             u'col_ref',
                                             u'col_refs',
                                             u'condition',
                                             u'conditions',
                                             u'conj',
                                             u'distinct',
                                             u'in_clause',
                                             u'number',
                                             u'pos_value',
                                             u'query',
                                             u'select_results',
                                             u'statement',
                                             u'string',
                                             u'table_name',
                                             u'table_refs',
                                             u'ternaryexpr',
                                             u'value',
                                             u'where_clause'])

        assert set(valid_actions[u'statement']) == set([u'statement -> [query, ";"]'])
        assert set(valid_actions[u'query']) ==\
                set([u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 u'where_clause, ")"]',
                 u'query -> ["SELECT", distinct, select_results, "FROM", table_refs, '
                 u'where_clause]'])
        assert set(valid_actions[u'select_results']) ==\
                set([u'select_results -> [agg]', u'select_results -> [col_refs]'])
        assert set(valid_actions[u'agg']) ==\
                set([u'agg -> [agg_func, "(", col_ref, ")"]'])
        assert set(valid_actions[u'agg_func']) ==\
                set([u'agg_func -> ["COUNT"]',
                 u'agg_func -> ["MAX"]',
                 u'agg_func -> ["MIN"]'])
        assert set(valid_actions[u'col_refs']) ==\
                set([u'col_refs -> [col_ref]', u'col_refs -> [col_ref, ",", col_refs]'])
        assert set(valid_actions[u'table_refs']) ==\
                set([u'table_refs -> [table_name]', u'table_refs -> [table_name, ",", table_refs]'])
        assert set(valid_actions[u'where_clause']) ==\
                set([u'where_clause -> ["WHERE", "(", conditions, ")"]',
                 u'where_clause -> ["WHERE", conditions]'])
        assert set(valid_actions[u'conditions']) ==\
                set([u'conditions -> ["(", conditions, ")", conj, conditions]',
                 u'conditions -> ["(", conditions, ")"]',
                 u'conditions -> ["NOT", conditions]',
                 u'conditions -> [condition, conj, "(", conditions, ")"]',
                 u'conditions -> [condition, conj, conditions]',
                 u'conditions -> [condition]'])
        assert set(valid_actions[u'condition']) ==\
                set([u'condition -> [biexpr]',
                 u'condition -> [in_clause]',
                 u'condition -> [ternaryexpr]'])
        assert set(valid_actions[u'in_clause']) ==\
                set([u'in_clause -> [col_ref, "IN", query]'])
        assert set(valid_actions[u'biexpr']) ==\
                set([u'biexpr -> [col_ref, "LIKE", string]',
                 u'biexpr -> [col_ref, binaryop, value]',
                 u'biexpr -> [value, binaryop, value]'])
        assert set(valid_actions[u'binaryop']) ==\
                set([u'binaryop -> ["*"]',
                 u'binaryop -> ["+"]',
                 u'binaryop -> ["-"]',
                 u'binaryop -> ["/"]',
                 u'binaryop -> ["<"]',
                 u'binaryop -> ["<="]',
                 u'binaryop -> ["="]',
                 u'binaryop -> [">"]',
                 u'binaryop -> [">="]',
                 u'binaryop -> ["IS"]'])
        assert set(valid_actions[u'ternaryexpr']) ==\
                set([u'ternaryexpr -> [col_ref, "BETWEEN", value, "AND", value]',
                 u'ternaryexpr -> [col_ref, "NOT", "BETWEEN", value, "AND", value]'])
        assert set(valid_actions[u'value']) ==\
                set([u'value -> ["NOT", pos_value]',
                 u'value -> [pos_value]'])
        assert set(valid_actions[u'pos_value']) ==\
                set([u'pos_value -> ["ALL", query]',
                 u'pos_value -> ["ANY", query]',
                 u'pos_value -> ["NULL"]',
                 u'pos_value -> [agg_results]',
                 u'pos_value -> [boolean]',
                 u'pos_value -> [col_ref]',
                 u'pos_value -> [number]',
                 u'pos_value -> [string]'])
        assert set(valid_actions[u'agg_results']) ==\
                set([(u'agg_results -> ["(", "SELECT", distinct, agg, "FROM", table_name, '
                  u'where_clause, ")"]'),
                 u'agg_results -> ["SELECT", distinct, agg, "FROM", table_name, where_clause]'])
        assert set(valid_actions[u'boolean']) ==\
                set([u'boolean -> ["true"]', u'boolean -> ["false"]'])
        assert set(valid_actions[u'conj']) ==\
                set([u'conj -> ["OR"]', u'conj -> ["AND"]'])
        assert set(valid_actions[u'distinct']) ==\
               set([u'distinct -> [""]', u'distinct -> ["DISTINCT"]'])
        assert set(valid_actions[u'number']) ==\
                set([u'number -> ["0"]',
                 u'number -> ["1"]'])
        assert set(valid_actions[u'string']) == set()
        assert set(valid_actions[u'col_ref']) ==\
                set([u'col_ref -> ["*"]',
                 u'col_ref -> ["aircraft", ".", "aircraft_code"]',
                 u'col_ref -> ["aircraft", ".", "aircraft_description"]',
                 u'col_ref -> ["aircraft", ".", "basic_type"]',
                 u'col_ref -> ["aircraft", ".", "manufacturer"]',
                 u'col_ref -> ["aircraft", ".", "pressurized"]',
                 u'col_ref -> ["aircraft", ".", "propulsion"]',
                 u'col_ref -> ["aircraft", ".", "wide_body"]',
                 u'col_ref -> ["airline", ".", "airline_code"]',
                 u'col_ref -> ["airline", ".", "airline_name"]',
                 u'col_ref -> ["airport", ".", "airport_code"]',
                 u'col_ref -> ["airport", ".", "airport_location"]',
                 u'col_ref -> ["airport", ".", "airport_name"]',
                 u'col_ref -> ["airport", ".", "country_name"]',
                 u'col_ref -> ["airport", ".", "minimum_connect_time"]',
                 u'col_ref -> ["airport", ".", "state_code"]',
                 u'col_ref -> ["airport", ".", "time_zone_code"]',
                 u'col_ref -> ["airport_service", ".", "airport_code"]',
                 u'col_ref -> ["airport_service", ".", "city_code"]',
                 u'col_ref -> ["airport_service", ".", "direction"]',
                 u'col_ref -> ["airport_service", ".", "miles_distant"]',
                 u'col_ref -> ["airport_service", ".", "minutes_distant"]',
                 u'col_ref -> ["city", ".", "city_code"]',
                 u'col_ref -> ["city", ".", "city_name"]',
                 u'col_ref -> ["city", ".", "country_name"]',
                 u'col_ref -> ["city", ".", "state_code"]',
                 u'col_ref -> ["city", ".", "time_zone_code"]',
                 u'col_ref -> ["class_of_service", ".", "booking_class"]',
                 u'col_ref -> ["class_of_service", ".", "class_description"]',
                 u'col_ref -> ["class_of_service", ".", "rank"]',
                 u'col_ref -> ["date_day", ".", "day_name"]',
                 u'col_ref -> ["date_day", ".", "day_number"]',
                 u'col_ref -> ["date_day", ".", "month_number"]',
                 u'col_ref -> ["date_day", ".", "year"]',
                 u'col_ref -> ["days", ".", "day_name"]',
                 u'col_ref -> ["days", ".", "days_code"]',
                 u'col_ref -> ["equipment_sequence", ".", "aircraft_code"]',
                 u'col_ref -> ["equipment_sequence", ".", "aircraft_code_sequence"]',
                 u'col_ref -> ["fare", ".", "fare_airline"]',
                 u'col_ref -> ["fare", ".", "fare_basis_code"]',
                 u'col_ref -> ["fare", ".", "fare_id"]',
                 u'col_ref -> ["fare", ".", "from_airport"]',
                 u'col_ref -> ["fare", ".", "one_direction_cost"]',
                 u'col_ref -> ["fare", ".", "restriction_code"]',
                 u'col_ref -> ["fare", ".", "round_trip_cost"]',
                 u'col_ref -> ["fare", ".", "round_trip_required"]',
                 u'col_ref -> ["fare", ".", "to_airport"]',
                 u'col_ref -> ["fare_basis", ".", "basis_days"]',
                 u'col_ref -> ["fare_basis", ".", "booking_class"]',
                 u'col_ref -> ["fare_basis", ".", "class_type"]',
                 u'col_ref -> ["fare_basis", ".", "discounted"]',
                 u'col_ref -> ["fare_basis", ".", "economy"]',
                 u'col_ref -> ["fare_basis", ".", "fare_basis_code"]',
                 u'col_ref -> ["fare_basis", ".", "night"]',
                 u'col_ref -> ["fare_basis", ".", "premium"]',
                 u'col_ref -> ["fare_basis", ".", "season"]',
                 u'col_ref -> ["flight", ".", "aircraft_code_sequence"]',
                 u'col_ref -> ["flight", ".", "airline_code"]',
                 u'col_ref -> ["flight", ".", "airline_flight"]',
                 u'col_ref -> ["flight", ".", "arrival_time"]',
                 u'col_ref -> ["flight", ".", "connections"]',
                 u'col_ref -> ["flight", ".", "departure_time"]',
                 u'col_ref -> ["flight", ".", "dual_carrier"]',
                 u'col_ref -> ["flight", ".", "flight_days"]',
                 u'col_ref -> ["flight", ".", "flight_id"]',
                 u'col_ref -> ["flight", ".", "flight_number"]',
                 u'col_ref -> ["flight", ".", "from_airport"]',
                 u'col_ref -> ["flight", ".", "meal_code"]',
                 u'col_ref -> ["flight", ".", "stops"]',
                 u'col_ref -> ["flight", ".", "time_elapsed"]',
                 u'col_ref -> ["flight", ".", "to_airport"]',
                 u'col_ref -> ["flight_fare", ".", "fare_id"]',
                 u'col_ref -> ["flight_fare", ".", "flight_id"]',
                 u'col_ref -> ["flight_leg", ".", "flight_id"]',
                 u'col_ref -> ["flight_leg", ".", "leg_flight"]',
                 u'col_ref -> ["flight_leg", ".", "leg_number"]',
                 u'col_ref -> ["flight_stop", ".", "arrival_airline"]',
                 u'col_ref -> ["flight_stop", ".", "arrival_flight_number"]',
                 u'col_ref -> ["flight_stop", ".", "arrival_time"]',
                 u'col_ref -> ["flight_stop", ".", "departure_airline"]',
                 u'col_ref -> ["flight_stop", ".", "departure_flight_number"]',
                 u'col_ref -> ["flight_stop", ".", "departure_time"]',
                 u'col_ref -> ["flight_stop", ".", "flight_id"]',
                 u'col_ref -> ["flight_stop", ".", "stop_airport"]',
                 u'col_ref -> ["flight_stop", ".", "stop_days"]',
                 u'col_ref -> ["flight_stop", ".", "stop_number"]',
                 u'col_ref -> ["flight_stop", ".", "stop_time"]',
                 u'col_ref -> ["food_service", ".", "compartment"]',
                 u'col_ref -> ["food_service", ".", "meal_code"]',
                 u'col_ref -> ["food_service", ".", "meal_description"]',
                 u'col_ref -> ["food_service", ".", "meal_number"]',
                 u'col_ref -> ["ground_service", ".", "airport_code"]',
                 u'col_ref -> ["ground_service", ".", "city_code"]',
                 u'col_ref -> ["ground_service", ".", "ground_fare"]',
                 u'col_ref -> ["ground_service", ".", "transport_type"]',
                 u'col_ref -> ["month", ".", "month_name"]',
                 u'col_ref -> ["month", ".", "month_number"]',
                 u'col_ref -> ["restriction", ".", "advance_purchase"]',
                 u'col_ref -> ["restriction", ".", "application"]',
                 u'col_ref -> ["restriction", ".", "maximum_stay"]',
                 u'col_ref -> ["restriction", ".", "minimum_stay"]',
                 u'col_ref -> ["restriction", ".", "no_discounts"]',
                 u'col_ref -> ["restriction", ".", "restriction_code"]',
                 u'col_ref -> ["restriction", ".", "saturday_stay_required"]',
                 u'col_ref -> ["restriction", ".", "stopovers"]',
                 u'col_ref -> ["state", ".", "country_name"]',
                 u'col_ref -> ["state", ".", "state_code"]',
                 u'col_ref -> ["state", ".", "state_name"]'])

        assert set(valid_actions[u'table_name']) ==\
                set([u'table_name -> ["aircraft"]',
                 u'table_name -> ["airline"]',
                 u'table_name -> ["airport"]',
                 u'table_name -> ["airport_service"]',
                 u'table_name -> ["city"]',
                 u'table_name -> ["class_of_service"]',
                 u'table_name -> ["date_day"]',
                 u'table_name -> ["days"]',
                 u'table_name -> ["equipment_sequence"]',
                 u'table_name -> ["fare"]',
                 u'table_name -> ["fare_basis"]',
                 u'table_name -> ["flight"]',
                 u'table_name -> ["flight_fare"]',
                 u'table_name -> ["flight_leg"]',
                 u'table_name -> ["flight_stop"]',
                 u'table_name -> ["food_service"]',
                 u'table_name -> ["ground_service"]',
                 u'table_name -> ["month"]',
                 u'table_name -> ["restriction"]',
                 u'table_name -> ["state"]'])

    def test_atis_local_actions(self): # pylint: disable=no-self-use
        # Check if the triggers activate correcty
        world = AtisWorld([u"show me the flights from denver at 12 o'clock"])
        assert set(world.valid_actions[u'number']) ==\
            set([u'number -> ["0"]',
             u'number -> ["1"]',
             u'number -> ["1200"]',
             u'number -> ["2400"]'])

        assert set(world.valid_actions[u'string']) ==\
                set([u'string -> ["\'DENVER\'"]',
                 u'string -> ["\'DDEN\'"]',
                 u'string -> ["\'AT\'"]'])

        world = AtisWorld([u"show me the flights from denver at 12 o'clock",
                           u"show me the delta or united flights in afternoon"])

        assert set(world.valid_actions[u'number']) ==\
                set([u'number -> ["0"]',
                 u'number -> ["1"]',
                 u'number -> ["1800"]',
                 u'number -> ["1200"]',
                 u'number -> ["2400"]'])

        assert set(world.valid_actions[u'string']) ==\
                set([u'string -> ["\'DENVER\'"]',
                 u'string -> ["\'DDEN\'"]',
                 u'string -> ["\'AT\'"]',
                 u'string -> ["\'DL\'"]',
                 u'string -> ["\'UA\'"]',
                 u'string -> ["\'IN\'"]'])

        world = AtisWorld([u"i would like one coach reservation for \
                          may ninth from pittsburgh to atlanta leaving \
                          pittsburgh before 10 o'clock in morning 1991 \
                          august twenty sixth"])

        assert set(world.valid_actions[u'number']) ==\
                set([u'number -> ["0"]',
                 u'number -> ["1"]',
                 u'number -> ["9"]',
                 u'number -> ["8"]',
                 u'number -> ["6"]',
                 u'number -> ["5"]',
                 u'number -> ["26"]',
                 u'number -> ["2200"]',
                 u'number -> ["1991"]',
                 u'number -> ["1200"]',
                 u'number -> ["1000"]'])

        assert set(world.valid_actions[u'string']) ==\
                set([u'string -> ["\'COACH\'"]',
                 u'string -> ["\'PITTSBURGH\'"]',
                 u'string -> ["\'PIT\'"]',
                 u'string -> ["\'PPIT\'"]',
                 u'string -> ["\'ATLANTA\'"]',
                 u'string -> ["\'ATL\'"]',
                 u'string -> ["\'MATL\'"]',
                 u'string -> ["\'IN\'"]',
                 u'string -> ["\'MONDAY\'"]'])


    def test_atis_simple_action_sequence(self): # pylint: disable=no-self-use
        world = AtisWorld([(u"give me all flights from boston to "
                            u"philadelphia next week arriving after lunch")])
        action_sequence = world.get_action_sequence((u"(SELECT DISTINCT city . city_code , city . city_name "
                                                     u"FROM city WHERE ( city.city_name = 'BOSTON' ) );"))
        assert action_sequence == [u'statement -> [query, ";"]',
                                   u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   u'where_clause, ")"]',
                                   u'where_clause -> ["WHERE", "(", conditions, ")"]',
                                   u'conditions -> [condition]',
                                   u'condition -> [biexpr]',
                                   u'biexpr -> [col_ref, binaryop, value]',
                                   u'value -> [pos_value]',
                                   u'pos_value -> [string]',
                                   u'string -> ["\'BOSTON\'"]',
                                   u'binaryop -> ["="]',
                                   u'col_ref -> ["city", ".", "city_name"]',
                                   u'table_refs -> [table_name]',
                                   u'table_name -> ["city"]',
                                   u'select_results -> [col_refs]',
                                   u'col_refs -> [col_ref, ",", col_refs]',
                                   u'col_refs -> [col_ref]',
                                   u'col_ref -> ["city", ".", "city_name"]',
                                   u'col_ref -> ["city", ".", "city_code"]',
                                   u'distinct -> ["DISTINCT"]']

        action_sequence = world.get_action_sequence((u"( SELECT airport_service . airport_code "
                                                     u"FROM airport_service "
                                                     u"WHERE airport_service . city_code IN ( "
                                                     u"SELECT city . city_code FROM city "
                                                     u"WHERE city.city_name = 'BOSTON' ) ) ;"))

        assert action_sequence == [u'statement -> [query, ";"]',
                                   u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   u'where_clause, ")"]',
                                   u'where_clause -> ["WHERE", conditions]',
                                   u'conditions -> [condition]',
                                   u'condition -> [in_clause]',
                                   u'in_clause -> [col_ref, "IN", query]',
                                   u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   u'where_clause, ")"]',
                                   u'where_clause -> ["WHERE", conditions]',
                                   u'conditions -> [condition]',
                                   u'condition -> [biexpr]',
                                   u'biexpr -> [col_ref, binaryop, value]',
                                   u'value -> [pos_value]',
                                   u'pos_value -> [string]',
                                   u'string -> ["\'BOSTON\'"]',
                                   u'binaryop -> ["="]',
                                   u'col_ref -> ["city", ".", "city_name"]',
                                   u'table_refs -> [table_name]',
                                   u'table_name -> ["city"]',
                                   u'select_results -> [col_refs]',
                                   u'col_refs -> [col_ref]',
                                   u'col_ref -> ["city", ".", "city_code"]',
                                   u'distinct -> [""]',
                                   u'col_ref -> ["airport_service", ".", "city_code"]',
                                   u'table_refs -> [table_name]',
                                   u'table_name -> ["airport_service"]',
                                   u'select_results -> [col_refs]',
                                   u'col_refs -> [col_ref]',
                                   u'col_ref -> ["airport_service", ".", "airport_code"]',
                                   u'distinct -> [""]']

        action_sequence = world.get_action_sequence((u"( SELECT airport_service . airport_code "
                                                     u"FROM airport_service WHERE airport_service . city_code IN "
                                                     u"( SELECT city . city_code FROM city "
                                                     u"WHERE city.city_name = 'BOSTON' ) AND 1 = 1) ;"))

        assert action_sequence ==\
                [u'statement -> [query, ";"]',
                 u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 u'where_clause, ")"]',
                 u'where_clause -> ["WHERE", conditions]',
                 u'conditions -> [condition, conj, conditions]',
                 u'conditions -> [condition]',
                 u'condition -> [biexpr]',
                 u'biexpr -> [value, binaryop, value]',
                 u'value -> [pos_value]',
                 u'pos_value -> [number]',
                 u'number -> ["1"]',
                 u'binaryop -> ["="]',
                 u'value -> [pos_value]',
                 u'pos_value -> [number]',
                 u'number -> ["1"]',
                 u'conj -> ["AND"]',
                 u'condition -> [in_clause]',
                 u'in_clause -> [col_ref, "IN", query]',
                 u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 u'where_clause, ")"]',
                 u'where_clause -> ["WHERE", conditions]',
                 u'conditions -> [condition]',
                 u'condition -> [biexpr]',
                 u'biexpr -> [col_ref, binaryop, value]',
                 u'value -> [pos_value]',
                 u'pos_value -> [string]',
                 u'string -> ["\'BOSTON\'"]',
                 u'binaryop -> ["="]',
                 u'col_ref -> ["city", ".", "city_name"]',
                 u'table_refs -> [table_name]',
                 u'table_name -> ["city"]',
                 u'select_results -> [col_refs]',
                 u'col_refs -> [col_ref]',
                 u'col_ref -> ["city", ".", "city_code"]',
                 u'distinct -> [""]',
                 u'col_ref -> ["airport_service", ".", "city_code"]',
                 u'table_refs -> [table_name]',
                 u'table_name -> ["airport_service"]',
                 u'select_results -> [col_refs]',
                 u'col_refs -> [col_ref]',
                 u'col_ref -> ["airport_service", ".", "airport_code"]',
                 u'distinct -> [""]']

        world = AtisWorld([(u"give me all flights from boston to "
                            u"philadelphia next week arriving after lunch")])
        action_sequence = world.get_action_sequence((u"( SELECT DISTINCT flight.flight_id "
                                                     u"FROM flight WHERE "
                                                     u"( flight . from_airport IN "
                                                     u"( SELECT airport_service . airport_code "
                                                     u"FROM airport_service WHERE airport_service . city_code IN "
                                                     u"( SELECT city . city_code "
                                                     u"FROM city "
                                                     u"WHERE city.city_name = 'BOSTON' )))) ;"))

        assert action_sequence ==\
            [u'statement -> [query, ";"]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", "(", conditions, ")"]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [biexpr]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'value -> [pos_value]',
             u'pos_value -> [string]',
             u'string -> ["\'BOSTON\'"]',
             u'binaryop -> ["="]',
             u'col_ref -> ["city", ".", "city_name"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["city"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["city", ".", "city_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["airport_service", ".", "city_code"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["airport_service"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["airport_service", ".", "airport_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["flight", ".", "from_airport"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["flight"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["flight", ".", "flight_id"]',
             u'distinct -> ["DISTINCT"]']

    def test_atis_long_action_sequence(self): # pylint: disable=no-self-use
        world = AtisWorld([(u"what is the earliest flight in morning "
                            u"1993 june fourth from boston to pittsburgh")])
        action_sequence = world.get_action_sequence(u"( SELECT DISTINCT flight.flight_id "
                                                    u"FROM flight "
                                                    u"WHERE ( flight.departure_time = ( "
                                                    u"SELECT MIN ( flight.departure_time ) "
                                                    u"FROM flight "
                                                    u"WHERE ( flight.departure_time BETWEEN 0 AND 1200 AND "
                                                    u"( flight . from_airport IN ( "
                                                    u"SELECT airport_service . airport_code "
                                                    u"FROM airport_service WHERE airport_service . city_code "
                                                    u"IN ( "
                                                    u"SELECT city . city_code "
                                                    u"FROM city WHERE city.city_name = 'BOSTON' )) "
                                                    u"AND flight . to_airport IN ( "
                                                    u"SELECT airport_service . airport_code "
                                                    u"FROM airport_service "
                                                    u"WHERE airport_service . city_code IN ( "
                                                    u"SELECT city . city_code "
                                                    u"FROM city "
                                                    u"WHERE city.city_name = 'PITTSBURGH' )) ) ) ) AND "
                                                    u"( flight.departure_time BETWEEN 0 AND 1200 AND "
                                                    u"( flight . from_airport IN ( "
                                                    u"SELECT airport_service . airport_code "
                                                    u"FROM airport_service "
                                                    u"WHERE airport_service . city_code IN ( "
                                                    u"SELECT city . city_code "
                                                    u"FROM city WHERE city.city_name = 'BOSTON' )) "
                                                    u"AND flight . to_airport IN ( "
                                                    u"SELECT airport_service . airport_code "
                                                    u"FROM airport_service WHERE airport_service . city_code IN ( "
                                                    u"SELECT city . city_code "
                                                    u"FROM city "
                                                    u"WHERE city.city_name = 'PITTSBURGH' )) ) ) )   ) ;")
        assert action_sequence ==\
            [u'statement -> [query, ";"]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", "(", conditions, ")"]',
             u'conditions -> [condition, conj, conditions]',
             u'conditions -> ["(", conditions, ")"]',
             u'conditions -> [condition, conj, conditions]',
             u'conditions -> ["(", conditions, ")"]',
             u'conditions -> [condition, conj, conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [biexpr]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'value -> [pos_value]',
             u'pos_value -> [string]',
             u'string -> ["\'PITTSBURGH\'"]',
             u'binaryop -> ["="]',
             u'col_ref -> ["city", ".", "city_name"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["city"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["city", ".", "city_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["airport_service", ".", "city_code"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["airport_service"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["airport_service", ".", "airport_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["flight", ".", "to_airport"]',
             u'conj -> ["AND"]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [biexpr]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'value -> [pos_value]',
             u'pos_value -> [string]',
             u'string -> ["\'BOSTON\'"]',
             u'binaryop -> ["="]',
             u'col_ref -> ["city", ".", "city_name"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["city"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["city", ".", "city_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["airport_service", ".", "city_code"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["airport_service"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["airport_service", ".", "airport_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["flight", ".", "from_airport"]',
             u'conj -> ["AND"]',
             u'condition -> [ternaryexpr]',
             u'ternaryexpr -> [col_ref, "BETWEEN", value, "AND", value]',
             u'value -> [pos_value]',
             u'pos_value -> [number]',
             u'number -> ["1200"]',
             u'value -> [pos_value]',
             u'pos_value -> [number]',
             u'number -> ["0"]',
             u'col_ref -> ["flight", ".", "departure_time"]',
             u'conj -> ["AND"]',
             u'condition -> [biexpr]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'value -> [pos_value]',
             u'pos_value -> [agg_results]',
             u'agg_results -> ["(", "SELECT", distinct, agg, "FROM", table_name, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", "(", conditions, ")"]',
             u'conditions -> [condition, conj, conditions]',
             u'conditions -> ["(", conditions, ")"]',
             u'conditions -> [condition, conj, conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [biexpr]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'value -> [pos_value]',
             u'pos_value -> [string]',
             u'string -> ["\'PITTSBURGH\'"]',
             u'binaryop -> ["="]',
             u'col_ref -> ["city", ".", "city_name"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["city"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["city", ".", "city_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["airport_service", ".", "city_code"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["airport_service"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["airport_service", ".", "airport_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["flight", ".", "to_airport"]',
             u'conj -> ["AND"]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [in_clause]',
             u'in_clause -> [col_ref, "IN", query]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'where_clause -> ["WHERE", conditions]',
             u'conditions -> [condition]',
             u'condition -> [biexpr]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'value -> [pos_value]',
             u'pos_value -> [string]',
             u'string -> ["\'BOSTON\'"]',
             u'binaryop -> ["="]',
             u'col_ref -> ["city", ".", "city_name"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["city"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["city", ".", "city_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["airport_service", ".", "city_code"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["airport_service"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["airport_service", ".", "airport_code"]',
             u'distinct -> [""]',
             u'col_ref -> ["flight", ".", "from_airport"]',
             u'conj -> ["AND"]',
             u'condition -> [ternaryexpr]',
             u'ternaryexpr -> [col_ref, "BETWEEN", value, "AND", value]',
             u'value -> [pos_value]',
             u'pos_value -> [number]',
             u'number -> ["1200"]',
             u'value -> [pos_value]',
             u'pos_value -> [number]',
             u'number -> ["0"]',
             u'col_ref -> ["flight", ".", "departure_time"]',
             u'table_name -> ["flight"]',
             u'agg -> [agg_func, "(", col_ref, ")"]',
             u'col_ref -> ["flight", ".", "departure_time"]',
             u'agg_func -> ["MIN"]',
             u'distinct -> [""]',
             u'binaryop -> ["="]',
             u'col_ref -> ["flight", ".", "departure_time"]',
             u'table_refs -> [table_name]',
             u'table_name -> ["flight"]',
             u'select_results -> [col_refs]',
             u'col_refs -> [col_ref]',
             u'col_ref -> ["flight", ".", "flight_id"]',
             u'distinct -> ["DISTINCT"]']


    def test_atis_from_json(self):
        line = json.loads(self.data[0])
        for utterance_idx in range(len(line[u'interaction'])):
            world = AtisWorld([interaction[u'utterance'] for
                               interaction in line[u'interaction'][:utterance_idx+1]])
            action_sequence = world.get_action_sequence(line[u'interaction'][utterance_idx][u'sql'])
            assert action_sequence is not None

    def test_all_possible_actions(self): # pylint: disable=no-self-use
        world = AtisWorld([(u"give me all flights from boston to "
                            u"philadelphia next week arriving after lunch")])
        possible_actions = world.all_possible_actions()

        assert possible_actions ==\
            [u'agg -> [agg_func, "(", col_ref, ")"]',
             u'agg_func -> ["COUNT"]',
             u'agg_func -> ["MAX"]',
             u'agg_func -> ["MIN"]',
             u'agg_results -> ["(", "SELECT", distinct, agg, "FROM", table_name, '
             u'where_clause, ")"]',
             u'agg_results -> ["SELECT", distinct, agg, "FROM", table_name, where_clause]',
             u'biexpr -> [col_ref, "LIKE", string]',
             u'biexpr -> [col_ref, binaryop, value]',
             u'biexpr -> [value, binaryop, value]',
             u'binaryop -> ["*"]',
             u'binaryop -> ["+"]',
             u'binaryop -> ["-"]',
             u'binaryop -> ["/"]',
             u'binaryop -> ["<"]',
             u'binaryop -> ["<="]',
             u'binaryop -> ["="]',
             u'binaryop -> [">"]',
             u'binaryop -> [">="]',
             u'binaryop -> ["IS"]',
             u'boolean -> ["false"]',
             u'boolean -> ["true"]',
             u'col_ref -> ["*"]',
             u'col_ref -> ["aircraft", ".", "aircraft_code"]',
             u'col_ref -> ["aircraft", ".", "aircraft_description"]',
             u'col_ref -> ["aircraft", ".", "basic_type"]',
             u'col_ref -> ["aircraft", ".", "manufacturer"]',
             u'col_ref -> ["aircraft", ".", "pressurized"]',
             u'col_ref -> ["aircraft", ".", "propulsion"]',
             u'col_ref -> ["aircraft", ".", "wide_body"]',
             u'col_ref -> ["airline", ".", "airline_code"]',
             u'col_ref -> ["airline", ".", "airline_name"]',
             u'col_ref -> ["airport", ".", "airport_code"]',
             u'col_ref -> ["airport", ".", "airport_location"]',
             u'col_ref -> ["airport", ".", "airport_name"]',
             u'col_ref -> ["airport", ".", "country_name"]',
             u'col_ref -> ["airport", ".", "minimum_connect_time"]',
             u'col_ref -> ["airport", ".", "state_code"]',
             u'col_ref -> ["airport", ".", "time_zone_code"]',
             u'col_ref -> ["airport_service", ".", "airport_code"]',
             u'col_ref -> ["airport_service", ".", "city_code"]',
             u'col_ref -> ["airport_service", ".", "direction"]',
             u'col_ref -> ["airport_service", ".", "miles_distant"]',
             u'col_ref -> ["airport_service", ".", "minutes_distant"]',
             u'col_ref -> ["city", ".", "city_code"]',
             u'col_ref -> ["city", ".", "city_name"]',
             u'col_ref -> ["city", ".", "country_name"]',
             u'col_ref -> ["city", ".", "state_code"]',
             u'col_ref -> ["city", ".", "time_zone_code"]',
             u'col_ref -> ["class_of_service", ".", "booking_class"]',
             u'col_ref -> ["class_of_service", ".", "class_description"]',
             u'col_ref -> ["class_of_service", ".", "rank"]',
             u'col_ref -> ["date_day", ".", "day_name"]',
             u'col_ref -> ["date_day", ".", "day_number"]',
             u'col_ref -> ["date_day", ".", "month_number"]',
             u'col_ref -> ["date_day", ".", "year"]',
             u'col_ref -> ["days", ".", "day_name"]',
             u'col_ref -> ["days", ".", "days_code"]',
             u'col_ref -> ["equipment_sequence", ".", "aircraft_code"]',
             u'col_ref -> ["equipment_sequence", ".", "aircraft_code_sequence"]',
             u'col_ref -> ["fare", ".", "fare_airline"]',
             u'col_ref -> ["fare", ".", "fare_basis_code"]',
             u'col_ref -> ["fare", ".", "fare_id"]',
             u'col_ref -> ["fare", ".", "from_airport"]',
             u'col_ref -> ["fare", ".", "one_direction_cost"]',
             u'col_ref -> ["fare", ".", "restriction_code"]',
             u'col_ref -> ["fare", ".", "round_trip_cost"]',
             u'col_ref -> ["fare", ".", "round_trip_required"]',
             u'col_ref -> ["fare", ".", "to_airport"]',
             u'col_ref -> ["fare_basis", ".", "basis_days"]',
             u'col_ref -> ["fare_basis", ".", "booking_class"]',
             u'col_ref -> ["fare_basis", ".", "class_type"]',
             u'col_ref -> ["fare_basis", ".", "discounted"]',
             u'col_ref -> ["fare_basis", ".", "economy"]',
             u'col_ref -> ["fare_basis", ".", "fare_basis_code"]',
             u'col_ref -> ["fare_basis", ".", "night"]',
             u'col_ref -> ["fare_basis", ".", "premium"]',
             u'col_ref -> ["fare_basis", ".", "season"]',
             u'col_ref -> ["flight", ".", "aircraft_code_sequence"]',
             u'col_ref -> ["flight", ".", "airline_code"]',
             u'col_ref -> ["flight", ".", "airline_flight"]',
             u'col_ref -> ["flight", ".", "arrival_time"]',
             u'col_ref -> ["flight", ".", "connections"]',
             u'col_ref -> ["flight", ".", "departure_time"]',
             u'col_ref -> ["flight", ".", "dual_carrier"]',
             u'col_ref -> ["flight", ".", "flight_days"]',
             u'col_ref -> ["flight", ".", "flight_id"]',
             u'col_ref -> ["flight", ".", "flight_number"]',
             u'col_ref -> ["flight", ".", "from_airport"]',
             u'col_ref -> ["flight", ".", "meal_code"]',
             u'col_ref -> ["flight", ".", "stops"]',
             u'col_ref -> ["flight", ".", "time_elapsed"]',
             u'col_ref -> ["flight", ".", "to_airport"]',
             u'col_ref -> ["flight_fare", ".", "fare_id"]',
             u'col_ref -> ["flight_fare", ".", "flight_id"]',
             u'col_ref -> ["flight_leg", ".", "flight_id"]',
             u'col_ref -> ["flight_leg", ".", "leg_flight"]',
             u'col_ref -> ["flight_leg", ".", "leg_number"]',
             u'col_ref -> ["flight_stop", ".", "arrival_airline"]',
             u'col_ref -> ["flight_stop", ".", "arrival_flight_number"]',
             u'col_ref -> ["flight_stop", ".", "arrival_time"]',
             u'col_ref -> ["flight_stop", ".", "departure_airline"]',
             u'col_ref -> ["flight_stop", ".", "departure_flight_number"]',
             u'col_ref -> ["flight_stop", ".", "departure_time"]',
             u'col_ref -> ["flight_stop", ".", "flight_id"]',
             u'col_ref -> ["flight_stop", ".", "stop_airport"]',
             u'col_ref -> ["flight_stop", ".", "stop_days"]',
             u'col_ref -> ["flight_stop", ".", "stop_number"]',
             u'col_ref -> ["flight_stop", ".", "stop_time"]',
             u'col_ref -> ["food_service", ".", "compartment"]',
             u'col_ref -> ["food_service", ".", "meal_code"]',
             u'col_ref -> ["food_service", ".", "meal_description"]',
             u'col_ref -> ["food_service", ".", "meal_number"]',
             u'col_ref -> ["ground_service", ".", "airport_code"]',
             u'col_ref -> ["ground_service", ".", "city_code"]',
             u'col_ref -> ["ground_service", ".", "ground_fare"]',
             u'col_ref -> ["ground_service", ".", "transport_type"]',
             u'col_ref -> ["month", ".", "month_name"]',
             u'col_ref -> ["month", ".", "month_number"]',
             u'col_ref -> ["restriction", ".", "advance_purchase"]',
             u'col_ref -> ["restriction", ".", "application"]',
             u'col_ref -> ["restriction", ".", "maximum_stay"]',
             u'col_ref -> ["restriction", ".", "minimum_stay"]',
             u'col_ref -> ["restriction", ".", "no_discounts"]',
             u'col_ref -> ["restriction", ".", "restriction_code"]',
             u'col_ref -> ["restriction", ".", "saturday_stay_required"]',
             u'col_ref -> ["restriction", ".", "stopovers"]',
             u'col_ref -> ["state", ".", "country_name"]',
             u'col_ref -> ["state", ".", "state_code"]',
             u'col_ref -> ["state", ".", "state_name"]',
             u'col_refs -> [col_ref, ",", col_refs]',
             u'col_refs -> [col_ref]',
             u'condition -> [biexpr]',
             u'condition -> [in_clause]',
             u'condition -> [ternaryexpr]',
             u'conditions -> ["(", conditions, ")", conj, conditions]',
             u'conditions -> ["(", conditions, ")"]',
             u'conditions -> ["NOT", conditions]',
             u'conditions -> [condition, conj, "(", conditions, ")"]',
             u'conditions -> [condition, conj, conditions]',
             u'conditions -> [condition]',
             u'conj -> ["AND"]',
             u'conj -> ["OR"]',
             u'distinct -> [""]',
             u'distinct -> ["DISTINCT"]',
             u'in_clause -> [col_ref, "IN", query]',
             u'number -> ["0"]',
             u'number -> ["1"]',
             u'number -> ["1200"]',
             u'number -> ["1400"]',
             u'number -> ["1800"]',
             u'pos_value -> ["ALL", query]',
             u'pos_value -> ["ANY", query]',
             u'pos_value -> ["NULL"]',
             u'pos_value -> [agg_results]',
             u'pos_value -> [boolean]',
             u'pos_value -> [col_ref]',
             u'pos_value -> [number]',
             u'pos_value -> [string]',
             u'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause, ")"]',
             u'query -> ["SELECT", distinct, select_results, "FROM", table_refs, '
             u'where_clause]',
             u'select_results -> [agg]',
             u'select_results -> [col_refs]',
             u'statement -> [query, ";"]',
             u'string -> ["\'BBOS\'"]',
             u'string -> ["\'BOS\'"]',
             u'string -> ["\'BOSTON\'"]',
             u'string -> ["\'LUNCH\'"]',
             u'string -> ["\'PHILADELPHIA\'"]',
             u'string -> ["\'PHL\'"]',
             u'string -> ["\'PPHL\'"]',
             u'table_name -> ["aircraft"]',
             u'table_name -> ["airline"]',
             u'table_name -> ["airport"]',
             u'table_name -> ["airport_service"]',
             u'table_name -> ["city"]',
             u'table_name -> ["class_of_service"]',
             u'table_name -> ["date_day"]',
             u'table_name -> ["days"]',
             u'table_name -> ["equipment_sequence"]',
             u'table_name -> ["fare"]',
             u'table_name -> ["fare_basis"]',
             u'table_name -> ["flight"]',
             u'table_name -> ["flight_fare"]',
             u'table_name -> ["flight_leg"]',
             u'table_name -> ["flight_stop"]',
             u'table_name -> ["food_service"]',
             u'table_name -> ["ground_service"]',
             u'table_name -> ["month"]',
             u'table_name -> ["restriction"]',
             u'table_name -> ["state"]',
             u'table_refs -> [table_name, ",", table_refs]',
             u'table_refs -> [table_name]',
             u'ternaryexpr -> [col_ref, "BETWEEN", value, "AND", value]',
             u'ternaryexpr -> [col_ref, "NOT", "BETWEEN", value, "AND", value]',
             u'value -> ["NOT", pos_value]',
             u'value -> [pos_value]',
             u'where_clause -> ["WHERE", "(", conditions, ")"]',
             u'where_clause -> ["WHERE", conditions]']
