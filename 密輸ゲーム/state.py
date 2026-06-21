from dataclasses import dataclass,field
from typing import List
from phase import Phase

@dataclass
class PlayerState:
    id :int
    team:int
    personal_account:int = 100

@dataclass
class TeamState:
    team:int
    members:List[int] = field(default_factory=list)
    representative:int = -1

@dataclass
class RoundInfo:
    smuggled_amount: int = 0
    declared_amount: int = 0
    inspected:bool = False
    doubt_amount:int = 0
    success:bool = False
    gain_amount:int = 0
    stolen_amount:int = 0

@dataclass
class GameState:
    phase:Phase = Phase.VOTE
    round:int = 1
    turn:int = 0
    max_rounds:int = 30
    players:List[PlayerState] = field(default_factory=list)
    smuggler:TeamState = field(default_factory=lambda:TeamState(team=0))
    inspector:TeamState = field(default_factory=lambda:TeamState(team=1))
    foreign1_account:int = 0
    foreign2_account:int = 0
    current:RoundInfo = field(default_factory=RoundInfo)

