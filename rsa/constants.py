# Constants
class severity:
    LOW = 0
    HIGH = 1    

class outcome:
    UNKNOWN = -1
    FATAL = 1
    SERIOUS = 2
    MINOR = 3
    NOT_INJURED = 4

class vehicle_type:
    BICYCLE = 1
    MOTORCYCLE = 2
    CAR = 3
    GOODS_VEHICLE = 4
    BUS = 5
    OTHER = 6

class circumstance:
    PEDESTRIAN = 1
    SINGLE_VEHICLE_ONLY = 2
    HEAD_ON_CONFLICT = 3
    HEAD_ON_RIGHT_TURN = 4
    ANGLE_BOTH_STRAIGHT = 5
    ANGLE_RIGHT_TURN = 6
    REAR_END_STRAIGHT = 7
    REAR_END_RIGHT_TURN = 8
    REAR_END_LEFT_TURN = 9
    OTHER = 10

class age_groups:
    _0_9 = 1
    _10_14 = 2
    _15_17 = 3
    _18_20 = 4
    _21_24 = 5
    _25_34 = 6
    _35_44 = 7
    _45_54 = 8
    _55_64 = 9
    _65_And_Over = 10
    Unknown = 99

# Dictionary helpers
counties = { 6: "Dublin", 9: "Kildare", 17: "Meath", 26: "Wicklow"}

# https://www.rsa.ie/docs/default-source/default-document-library/road-casualties-and-collisions-in-ireland-2014---tables.pdf?Status=Master&sfvrsn=2a59a650_3
ageGroups = {
    1: "G1: 0-9",
    2: "G2: 10-14",
    3: "G3: 15-17",
    4: "G4: 18-20",
    5: "G5: 21-24",
    6: "G6: 25-34",
    7: "G7: 35-44",
    8: "G8: 45-54",
    9: "G9: 55-64",
    65: "G11: 65 and Over",
    99: "Unknown",
}

gender = {
    0: "Unknown", 
    1: "Male", 
    2: "Female"};

dayOfWeek = {
    1: "Sunday", 
    2: "Monday", 
    3: "Tuesday", 
    4: "Wednesday", 
    5: "Thursday", 
    6: "Friday",
    7: "Saturday"};

dayOfWeekShort= {
    1: "S", 
    2: "M", 
    3: "T", 
    4: "W", 
    5: "T", 
    6: "F", 
    7: "S"};

hours = {
    1: "0700-1000", 
    2: "1000-1600", 
    3: "1600-1900", 
    4: "1900-2300", 
    5: "2300-0300", 
    6: "0300-0700"};

hoursLong = {
    1: "7am to 10am", 
    2: "10am to 4pm", 
    3: "4pm to 7pm", 
    4: "7pm to 11pm", 
    5: "11pm to 3am",
    6: "3am to 7am"
    };

fireBrigadeStations = {
    0:'Tara Street',
    1:'Donnybrook',
    2:'Dolphins Barn',
    3:'Phibsboro', 
    4:'North Strand',
    5:'Finglas', 
    6:'Kilbarrack', 
    7:'Tallaght',
    8:'Rathfarnham',
    9:'Blanchardstown',
    10:'Skerries',
    11:'Balbriggan',
    12:'Dun Laoghaire',
    13:'Swords'
}

class feature_names:
    """ RSA features names
    """    
    ID = 'id'
    GPS = 'gps'                     # gps coordinates lat,long
    LATITUDE = 'latitude'           # latitude
    LONGITUDE = 'longitude'         # longitude
    SPLIMIT = 'splimit'             # Speed limit
    OUTCOME = 'outcome'             # 1 Fatal, 2 Serious, 3 Minor
    OUTCOME_CALCULATED = 'outcome_calculated'
    GENDER =  'gender'              # gender
    AGE = 'age'                     # age
    COUNTY = 'county'               # county
    YEAR = 'year'                   # Year
    WEEKDAY = 'weekday'             # Week day
    HOUR = 'hour'                   # Hour period
    CIRCUMSTANCES = 'circumstances' # Circumstances
    VEHICLE = 'vehicle'             # Vehicle
    VEHICLE_TYPE = 'vehicle_type'   # Vehicle Type
    VEHICLE_TYPE2 = 'vehicle_type2' # Vehicle Type 2
    
    # Safety Index
    RADIUS_500M = 'radius_500m'
    
    # No. casualties - total = no_fatal + no_serious + no_minor
    NO_SERIOUS = 'no_serious'       # No serious casualties
    NO_MINOR = 'no_minor'           # No minor casualties
    NO_FATAL = 'no_fatal'           # No fatal casualties
    NO_NOTINJURED = 'no_notinjured' # No not injured casualties
    NO_UNKONWN = 'no_unknown'       # No Unkonwn casualties
    IS_FATAL = 'is_fatal'           # Is fatal accident
    SEVERITY = 'severity'           # Severity: Low (Minor) / High (Serious / Fatal)
    FBS_STATION = 'fbs_station'     # Firebrigade and Ambulance
     
    # rf ri - Car, Pedal Cycle Users, Goods Vehicle User, Pedestrians, PSV Users (Public Service Vehicles), Motor Cycle Users, Unknown
    CAR_RF = 'carrf'                 # car rf 
    CAR_RI = 'carri'                 # car ri
    PCYC_RF = 'pcycrf'               # Pedal Cycle rf 
    PCYC_RI = 'pcycri'               # Pedal Cycle ri
    GOODS_RF = 'goodsrf'             # Goods vehicle rf
    GOODS_RI = 'goodsri'             # Goods vehicle ri
    PED_RF = 'pedrf'                 # Pedestrian rf
    PED_RI = 'pedri'                 # Pedestrian ri
    PSV_RF = 'psvrf'                 # Public Service Vehicles rf
    PSV_RI = 'psvri'                 # Public Service Vehicles ri
    MCYC_RF = 'mcycrf'               # Motor cycle rf
    MCYC_RI = 'mcycri'               # Motor cycle ri
    UNKNOWN_RF = 'unknrf'            # Unkonwn rf
    UNKONWN_RI = 'unknri'            # Unkonwn ri
    OTHER_RF = 'otherrf'            # Other rf
    OTHER_RI = 'otherri'             # Other ri
    listForAnalisys = [ SPLIMIT, GENDER, AGE, YEAR, WEEKDAY, HOUR, CIRCUMSTANCES, VEHICLE_TYPE ]

class rsa_enums:
    severity = severity()
    outcome = outcome()
    circumstance = circumstance()
    vehicle_type = vehicle_type()
    feature_names =  feature_names()
    age_groups = age_groups()

enums = rsa_enums()

circumstanceTypes = {
    enums.circumstance.PEDESTRIAN: "Pedestrian", 
    enums.circumstance.SINGLE_VEHICLE_ONLY: "Single vehicle only", 
    enums.circumstance.HEAD_ON_CONFLICT: "Head-on conflict", 
    enums.circumstance.HEAD_ON_RIGHT_TURN: "Head-on right turn", 
    enums.circumstance.ANGLE_BOTH_STRAIGHT: "Angle, both straight", 
    enums.circumstance.ANGLE_RIGHT_TURN: "Angle, right turn", 
    enums.circumstance.REAR_END_STRAIGHT: "Rear end, straight", 
    enums.circumstance.REAR_END_RIGHT_TURN: "Rear end, right turn", 
    enums.circumstance.REAR_END_LEFT_TURN: "Rear end, left turn", 
    enums.circumstance.OTHER: "Other"};

vehicleTypes = {
    enums.vehicle_type.BICYCLE: "Bicycle", 
    enums.vehicle_type.MOTORCYCLE: "Motorcycle", 
    enums.vehicle_type.CAR: "Car", 
    enums.vehicle_type.GOODS_VEHICLE: "Goods vehicle", 
    enums.vehicle_type.BUS: "Bus",
    enums.vehicle_type.OTHER: "Other"}

outcome = { 
    enums.outcome.FATAL: "Fatal",
    enums.outcome.SERIOUS: "Serious",
    enums.outcome.MINOR: "Minor"}

severityTypes = { 
    enums.severity.LOW: "Low",
    enums.severity.HIGH: "High"}

outcomeTypes =  {
    enums.outcome.FATAL: "Fatal", 
    enums.outcome.SERIOUS: "Serious", 
    enums.outcome.MINOR: "Minor",
    enums.outcome.NOT_INJURED: "Not Injured"
    };

class rsa_dictionaries:
    OUTCOME = outcome
    OUTCOME_TYPES = outcomeTypes
    SEVERITY= severityTypes
    VEHICLE_TYPES = vehicleTypes
    CIRCUMSTANCE_TYPES = circumstanceTypes
    COUNTIES = counties
    HOURS_LONG = hoursLong
    HOURS = hours
    DAY_OF_WEEK_SHORT = dayOfWeekShort
    DAY_OF_WEEK = dayOfWeek
    AGE_GROUPS = ageGroups
    GENDER = gender
    FIREBRIGADE_STATION = fireBrigadeStations

    
dictionaries = rsa_dictionaries()

__all__ = ['dictionaries','enums']