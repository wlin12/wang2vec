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

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
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
  char file_name[max_size], output_file[max_size];
  float len;
  long long words, size, a, b, c, d;
  float *M;
  char *vocab;
  char word[MAX_STRING];
  if (argc < 3) {
    printf("Usage: ./kmeans_txt <FILE>\nwhere FILE contains features\n <number_of_classes>");
    return 0;
  }
  strcpy(file_name, argv[1]);
  strcpy(output_file, argv[2]);
  int classes = atoi(argv[3]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  
  FILE *fo = fopen(output_file, "wb");
  
  ReadWord(word, f);
  words = atoi(word);
  ReadWord(word, f);
  size = atoi(word);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
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
  int clcn = classes, iter = 2, closeid;
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
    
    for (a = 0; a < classes; a++){
    	c = words;
    	if(a < classes-1) c = class_offsets[a+1];
    	b = class_offsets[a];
    	for(; b < c; b++){
    	    fprintf(fo, "%lld %s\n", a ,&vocab[class_words[b] * max_w]);
    	}
    }
     // Save the K-means classes
    //for (a = 0; a < words; a++) fprintf(fo, "%s %d\n", &vocab[a * max_w], cl[a]);
    free(centcn);
    free(cent);
    free(cl);
    free(M);
    free(vocab);
  return 0;
}
