import calc


if __name__ == '__main__':
    domains = ["books", "kitchen", "dvd", "electronics"]
    ngram = 2
    min_freq = 20
    for src in domains:
        for trg in domains:
            src_path = "data/" + src + "/" + src + "UN.txt"
            trg_path = "data/" + trg + "/" + trg + "UN.txt"
            a_distance = calc.calc_distance(src_path, trg_path, ngram, min_freq)
            print("the A-distance between {} and {} is {}".format(src, trg, round(a_distance, 3)))




