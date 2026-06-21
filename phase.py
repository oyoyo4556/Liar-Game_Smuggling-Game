from enum import IntEnum

class Phase(IntEnum):
    VOTE = 0
    SMUGGLE = 1
    INSPECT = 2
    DISTRIBUTE = 3
    PUBLIC_UPDATE = 4
    TERMINATED = 5