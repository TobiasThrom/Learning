from individual import Individual
import cv2
import numpy as np
import random

class Picture(Individual):

    mutationFactor = 10

    def __init__(self, size):
        self.size = size
        self.picture = self.generateRandomPicture(size)
        self.fitness = self.getFitness()

    def getFitness(self):
        return np.sum(self.picture)

    def generateRandomPicture(self, size):
        pic = np.random.random((size, size))
        pic = np.array(pic * 255, dtype=np.uint8)
        return pic

    def mutate(self):
        mutation = np.random.random((self.size, self.size))
        mutation = np.array((mutation-0.5)*self.mutationFactor)
        self.picture = np.array(self.picture+mutation, dtype=np.uint8)



    def newIndividual(self, Individual):
        split = np.random.randint(0, self.size*self.size)
        child = Picture(self.size)
        child.picture = self.picture.flatten()
        child.picture[split:] = Individual.picture.flatten()[split:]
        child.picture = child.picture.reshape(self.size, self.size)
        return child


    def getPicture(self):
        return self.picture

    def display(self):
        cv2.imshow('image', self.picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        