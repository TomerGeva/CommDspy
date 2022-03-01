from enum import Enum


class PrbsEnum(Enum):
    PRBS7  = 7
    PRBS9  = 9
    PRBS11 = 11
    PRBS13 = 13
    PRBS15 = 15
    PRBS31 = 31


class ConstellationEnum(Enum):
    NRZ  = 0
    OOK  = 1
    PAM4 = 2


class CodingEnum(Enum):
    UNCODED = 0
    GRAY    = 1
