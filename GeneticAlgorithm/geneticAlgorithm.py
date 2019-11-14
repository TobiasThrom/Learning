from population import Population
import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm():
    def __init__(self,popSize):
        self.population = Population(popSize)

    def run(self):
        #run for 100 generations
        #TODO add termination criterion if Fittest is greater than treshhold
        fittest,_ = self.population.getFittestIdx()
        ind = self.population.getIndividual(fittest)
        ind.display()
        generation = []
        fittestVal = []
        avgFitness = []
        while self.population.generation < 1000:
            gen,val,avg=self.population.getInfo()
            generation.append(gen)
            fittestVal.append(val)
            avgFitness.append(avg)
            self.population.repopulate()
            self.population.increaseGeneration()
        fittest,_ = self.population.getFittestIdx()
        ind2 = self.population.getIndividual(fittest)
        ind2.display()
        diff = ind2.getPicture().astype('int32')  - ind.getPicture().astype('int32')  
        print("Changes:")
        print(diff)
        print("Sum of changes:", diff.sum())
        plt.plot(fittestVal)
        plt.ylabel('Fitness of fittest Individual')
        plt.show()
        plt.plot(avgFitness)
        plt.ylabel('Average fitness of whole population')
        plt.show()
