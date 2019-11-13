from picture import Picture
class Population():

    def __init__(self,popSize):
        self.generation = 1
        self.popSize = popSize
        self.indiviuals = []
        for i in range(popSize):
            self.indiviuals.append(Picture(500))
    
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
