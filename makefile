CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -g

all: word2vec word2phrase distance word-analogy compute-accuracy distance_txt distance_fast kmeans_txt

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
distance_txt : distance_txt.c
	$(CC) distance_txt.c -o distance_txt $(CFLAGS)
distance_fast : distance_fast.c
	$(CC) distance_fast.c -o distance_fast $(CFLAGS)
kmeans_txt : kmeans_txt.c
	$(CC) kmeans_txt.c -o kmeans_txt $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy distance_txt kmeans_txt
