from picture import Picture

def main():
    p = Picture(5)
    p2 = Picture(5)
    print(p.getPicture())
    print(p2.getPicture())
    p3 = p.newIndividual(p2)
    print(p3.getPicture())


















if __name__ == "__main__":
    main()