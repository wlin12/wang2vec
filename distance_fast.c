//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

const long long max_size = 2000;         // max length of strings
const long long N = 10;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

#define MAX_STRING 100
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char *bestw[N];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], bestclasses[N], vec[max_size];
  int bestclasses_ids[N];
  long long words, size, a, b, c, d, e, cn, bi[100];
  float *M;
  char *vocab;
  char word[MAX_STRING];
  clock_t begin;
  if (argc < 2) {
    printf("Usage: ./kmeans_txt <FILE>\nwhere FILE contains features\n <number_of_classes>");
    return 0;
  }
  strcpy(file_name, argv[1]);
  int classes = atoi(argv[2]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  
  printf("reading data\n");
  ReadWord(word, f);
  words = atoi(word);
  ReadWord(word, f);
  size = atoi(word);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) {
        ReadWord(word,f); 
        M[a + b * size] = atof(word); 
    }
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  
  //run kmeans
  printf("running k-means with %i classes...\n",classes);
  int clcn = classes, iter = 10, closeid;
  int *centcn = (int *)malloc(classes * sizeof(int));
  int *cl = (int *)calloc(words, sizeof(int));
  float closev, x;
  float *cent = (float *)calloc(classes * size, sizeof(float));
  for (a = 0; a < words; a++) cl[a] = a % clcn;
  for (a = 0; a < iter; a++) {
    for (b = 0; b < clcn * size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < words; c++) {
      for (d = 0; d < size; d++) cent[size * cl[c] + d] += M[c * size + d];
      centcn[cl[c]]++;
    }
    for (b = 0; b < clcn; b++) {
      closev = 0;
      for (c = 0; c < size; c++) {
        cent[size * b + c] /= centcn[b];
        closev += cent[size * b + c] * cent[size * b + c];
      }
      closev = sqrt(closev);
      for (c = 0; c < size; c++) cent[size * b + c] /= closev;
    }
    for (c = 0; c < words; c++) {
      closev = -10;
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        for (b = 0; b < size; b++) x += cent[size * d + b] * M[c * size + b];
        if (x > closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }
  
  // build an array of words ordered by class and their offsets (index where each class starts)
  int class_words[words];
  int class_offsets[classes];
  for(a = 0; a < classes; a++) class_offsets[a]=0;
  for(a = 0; a < words; a++) class_offsets[cl[a]]++;
  for(a = 1; a < classes; a++) class_offsets[a] += class_offsets[a-1];
  for(a = 0; a < words; a++) class_words[--class_offsets[cl[a]]] = a;   
  
  //reading from input
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestclasses[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;
    begin = clock();
    
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    
    // find top N centroids
    for (a = 0; a < N; a++) bestclasses[a] = -1;
    for (a = 0; a < N; a++) bestclasses_ids[a] = -1;
    for (c = 0; c < classes; c++){
    	dist = 0;
    	for (a = 0; a < size; a++) dist += vec[a] * cent[a + size * c];
        for (a = 0; a < N; a++) {
        if (dist > bestclasses[a]) {
          	for(d = N - 1; d > a; d--){
          		bestclasses[d] = bestclasses[d-1];
          		bestclasses_ids[d] = bestclasses_ids[d-1];
          	}
          	bestclasses[a] = dist;
          	bestclasses_ids[a] = c;
          	break;
        }
    	}
    }
    
    // find top N words in the centroids
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (a = 0; a < N; a++){
   		c = words;
   		if(bestclasses_ids[a] < classes-1) c = class_offsets[bestclasses_ids[a]+1];
   		b = class_offsets[bestclasses_ids[a]];
   		for(; b < c; b++){
   			dist = 0;
            for (d = 0; d < size; d++) dist += vec[d] * M[d + class_words[b] * size];
            for (d = 0; d < N; d++){
            	if(dist > bestd[d]){
            		for (e = N -1; e > d; e--){
            			bestd[e] = bestd[e-1];
            			strcpy(bestw[e], bestw[e-1]);
            		}
            		bestd[d] = dist;
            		strcpy(bestw[d], &vocab[class_words[b] * max_w]);
            		break;
            	}
            }
   	    }
   	}
	for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);   
	printf("time spent = %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);
  }
  // Save the K-means classes

  free(centcn);
  free(cent);
  free(cl);
  
  //start running distance
  return 0;
}
