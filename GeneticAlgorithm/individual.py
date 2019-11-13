from abc import ABC, abstractmethod
 
class Individual(ABC):
     
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def getFitness(self):
        pass

    @abstractmethod
    def newIndividual(self, ind):
        pass