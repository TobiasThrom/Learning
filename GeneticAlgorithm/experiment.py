from picture import Picture

def main():
    p = Picture(5)
    print(p.getFitness())
    p.mutate()
    print(p.getFitness())


















if __name__ == "__main__":
    main()