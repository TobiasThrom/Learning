from individual import Individual
import cv2
import numpy as np
import random

class Picture(Individual):

    mutationFactor = 100

    def __init__(self, size):
        self.size = size
        self.picture = self.generateRandomPicture(size)
        self.fitness = self.getFitness()

    def getFitness(self):
        return np.sum(self.picture)

    def generateRandomPicture(self, size):
        pic = np.random.randint(2, size=(size, size))
        pic = np.array(pic * 255, dtype=np.uint8)
        return pic

    def mutate(self):
        self.picture = self.picture.flatten()
        for i in range(self.size*self.size):
            rand = np.random.randint(10)
            if rand == 0:
                if self.picture[i] == 0:
                    self.picture[i] = 255
                else:
                    self.picture[i] = 0
        self.picture = self.picture.reshape(self.size, self.size)




    def newIndividual(self, Individual):
        split = np.random.randint(0, self.size*self.size)
        child = Picture(self.size)
        child.picture = self.picture.flatten()
        child.picture[split:] = Individual.picture.flatten()[split:]
        child.picture = child.picture.reshape(self.size, self.size)
        child.mutate()
        child.fitness = child.getFitness()
        return child


    def getPicture(self):
        return self.picture

    def display(self):
        cv2.imshow('image', self.picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        