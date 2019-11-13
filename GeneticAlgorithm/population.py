from picture import Picture
class Population():

    def __init__(self,popSize):
        self.generation = 1
        self.popSize = popSize
        self.indiviuals = []
        for i in range(popSize):
            self.indiviuals.append(Picture(5))
    
    def getFittestIdx(self):
        fittest = self.indiviuals[0]
        fittestIdx = 0
        second = self.indiviuals[1]
        secondIdx = 1
        idx = 0
        for i in self.indiviuals:
            if (i.fitness > fittest.fitness): 
                second = fittest
                secondIdx = fittestIdx
                fittest = i
                fittestIdx = idx
            elif (i.fitness >second.fitness): 
                secondIdx = idx
            idx +=1
            
        return fittestIdx, secondIdx
    
    def getIndividual(self, idx):
        return self.indiviuals[idx]

    def getInfo(self):
        print("Generation:",self.generation)
        fittest, _ = self.getFittestIdx()
        print("Fittest:", self.getIndividual(fittest).getFitness())
        avgFitness = 0
        for i in self.indiviuals:
            avgFitness += i.getFitness()
        avgFitness = avgFitness/self.popSize
        print("Avg Fitness:", avgFitness)
    
    def increaseGeneration(self):
        self.generation +=1

    def repopulate(self):
        fittestIdx,secondIdx = self.getFittestIdx()
        fittest = self.getIndividual(fittestIdx)
        second = self.getIndividual(secondIdx)
        length = len(self.indiviuals) 
        for i in range(length-2):
            self.indiviuals[i] = fittest.newIndividual(second)
        self.indiviuals[length-2] = fittest
        self.indiviuals[length-1] = second

