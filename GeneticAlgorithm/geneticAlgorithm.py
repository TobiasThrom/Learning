from population import Population
import numpy as np

class GeneticAlgorithm():
    def __init__(self,popSize):
        self.population = Population(popSize)

    def run(self):
        #run for 100 generations
        #TODO add termination criterion if Fittest is greater than treshhold
        fittest,_ = self.population.getFittestIdx()
        ind = self.population.getIndividual(fittest)
        print(ind.getPicture())
        while self.population.generation < 100:
            self.population.getInfo()
            self.population.repopulate()
            self.population.increaseGeneration()
        fittest,_ = self.population.getFittestIdx()
        ind2 = self.population.getIndividual(fittest)
        print(ind2.getPicture())
        diff = ind2.getPicture().astype('int32')  - ind.getPicture().astype('int32')  
        print("Changes:")
        print(diff)
        print("Sum of changes:", diff.sum())