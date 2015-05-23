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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, type = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable, *tanhTable;
clock_t start;

real *syn1_window, *syn1neg_window;
int window_offset, window_layer_size;

int window_hidden_size = 500; 
real *syn_window_hidden, *syn_hidden_word, *syn_hidden_word_neg; 

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

//constrastive negative sampling
char negative_classes_file[MAX_STRING];
int *word_to_group;
int *group_to_table; //group_size*table_size
int class_number;

//char table 
int rep = 0;
#define C_MAX_CODE 65536
int c_state_size = 5;
int c_cell_size = 5;
int c_proj_size = 3;
int c_params_number;
int c_lstm_params_number;
real *c_lookup;

//char lstm params
real *f_init_state;
real *f_init_cell;
real *b_init_state;
real *b_init_cell;

real *f_b_params;

void printStates(real*states, int start){
	int s;
	printf("igate ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");
	printf("fgate ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");
	printf("c + tanh ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");
	printf("cgate ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");
	printf("ogate ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");
	printf("cgate + tanh ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");
	printf("state ");	
	for(s = 0; s < c_state_size; s++){ printf("%f ", states[start++]);} printf("\n");

}

void lstmForwardBlock(real *chars, int char_start, real*states, int next_start, int p){
	int i,s,si,sf,sc,sct,sh,sctt,so,s1=next_start;	
	int prev_cell_start = s1 - c_state_size*4;
	int prev_state_start = s1 - c_state_size;
	if(states[prev_cell_start]==0){
//		printf("crap! cell is zero\n");		
	}
	if(states[prev_state_start]==0){
//		printf("crap! state is zero\n");		
	}
	if(states[s1]!=0){
//		printf("crap! start not zero\n");
	}
	//igate
	si = s1;
	for(s = 0; s < c_state_size; s++){
		for(i = 0; i < c_proj_size; i++){
			states[s1]+=chars[char_start+i]*f_b_params[p++];
		}
		for(i = 0; i < c_cell_size; i++){
			states[s1]+=states[prev_cell_start+i]*f_b_params[p++];
		}
		for(i = 0; i < c_state_size; i++){
			states[s1]+=states[prev_state_start+i]*f_b_params[p++];
		}
		states[s1]+=f_b_params[p++];
		if(states[s1]>MAX_EXP){
			states[s1]=1;
		}
		else if(states[s1]<-MAX_EXP){
			states[s1]=0;
		}
		else{
			states[s1] = expTable[(int)((states[s1] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		}
		s1++;		
	}
	
	//fgate
	sf=s1;
	for(s = 0; s < c_state_size; s++){
		for(i = 0; i < c_proj_size; i++){
			states[s1]+=chars[char_start+i]*f_b_params[p++];
		}
		for(i = 0; i < c_cell_size; i++){
			states[s1]+=states[prev_cell_start+i]*f_b_params[p++];
		}
		for(i = 0; i < c_state_size; i++){
			states[s1]+=states[prev_state_start+i]*f_b_params[p++];
		}
		states[s1]+=f_b_params[p++];
		if(states[s1]>MAX_EXP){
			states[s1]=1;
		}
		else if(states[s1]<-MAX_EXP){
			states[s1]=0;
		}
		else{
			states[s1] = expTable[(int)((states[s1] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		}
		s1++;
	}
	
	//c + tanh
	sct=s1;
	for(s = 0; s < c_state_size; s++){
		for(i = 0; i < c_proj_size; i++){
			states[s1]+=chars[char_start+i]*f_b_params[p++];
		}
		for(i = 0; i < c_state_size; i++){
			states[s1]+=states[prev_state_start+i]*f_b_params[p++];
		}
		states[s1]+=f_b_params[p++];
		if(states[s1]>MAX_EXP){
			states[s1]=1;
		}
		else if(states[s1]<-MAX_EXP){
			states[s1]=-1;
		}
		else{
			states[s1] = tanhTable[(int)((states[s1] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		}
		s1++;
	}
	
	//cgate
	sc=s1;
	for(s = 0; s < c_state_size; s++){
		states[s1]+=states[sct+s]*states[si+s]+states[sf+s]*states[prev_cell_start+s];
		s1++;
	}
	
	//ogate
	so=s1;
	for(s = 0; s < c_state_size; s++){
		for(i = 0; i < c_proj_size; i++){
			states[s1]+=chars[char_start+i]*f_b_params[p++];
		}
		for(i = 0; i < c_cell_size; i++){
			states[s1]+=states[sc+s]*f_b_params[p++];
		}
		for(i = 0; i < c_state_size; i++){
			states[s1]+=states[prev_state_start+i]*f_b_params[p++];
		}
		states[s1]+=f_b_params[p++];
		if(states[s1]>MAX_EXP){
			states[s1]=1;
		}
		else if(states[s1]<-MAX_EXP){
			states[s1]=0;
		}
		else{
			states[s1] = expTable[(int)((states[s1] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		}
		s1++;		
	}
	
	//cgate + tan
	sctt = s1;
	for(s = 0; s < c_state_size; s++){
		if(states[sc+s]>MAX_EXP){
			states[s1]=1;
		}
		else if(states[sc+s]<-MAX_EXP){
			states[s1]=-1;
		}
		else{
			states[s1] = tanhTable[(int)((states[sc+s] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
		}
		s1++;
	}
	
	//next state
	sh = s1;
	if(states[s1]!=0){
		printf("crap! end not zero\n");
	}
	for(s = 0; s < c_state_size; s++){
		states[s1] = states[sctt+s] * states[so+s];
		s1++;
	}
	
	
}

void lstmBackwardBlock(real *chars, int char_start, real*states, int next_start, int pStart, real*chars_e, real*states_e, real*lstm_params_e){
	int p=pStart+c_lstm_params_number-1;
	int i,s,si,sf,sc,sct,sh,sctt,so,s1=next_start+c_state_size*7-1;
	int prev_cell_start = next_start - c_state_size*4;
	int prev_state_start = next_start - c_state_size;

	real e;
	si = next_start;
	sf = next_start + c_state_size;
	sct = next_start + c_state_size*2;
	sc = next_start + c_state_size*3;
	so = next_start + c_state_size*4;
	sctt = next_start + c_state_size*5;
	sh = next_start + c_state_size*6;
	
	//next state 
	for(s = c_state_size-1; s >= 0; s--){
		states_e[sctt+s] += states_e[s1]*states[so+s];
		states_e[so+s] += states_e[s1]*states[sctt+s];
		s1--;
	}	
	
	
	//cgate + tan	
	for(s = c_state_size-1; s >= 0; s--){
		states_e[sc+s] += states_e[s1]*(1-states[s1]*states[s1]);		
		s1--;
	}
	
	//ogate
	for(s = c_state_size-1; s >= 0; s--){
		e = states[s1]*(1-states[s1])*states_e[s1];
		for(i = c_proj_size-1; i >= 0; i--){
			chars_e[char_start+i] += e*f_b_params[p];
			lstm_params_e[p--] += e*chars[char_start+i];
		}

		for(i = c_cell_size-1; i >= 0; i--){
			states_e[sc+s]+=e*f_b_params[p];
			lstm_params_e[p--] += e*states[sc+s];
		}
		for(i = c_state_size-1; i >= 0; i--){
			states_e[prev_state_start+i] += e*f_b_params[p];
			lstm_params_e[p--] += e*states_e[prev_state_start+i];
		}
		lstm_params_e[p--]+=e;		
		s1--;		
	}
	
	//cgate
	for(s = c_state_size-1; s >= 0; s--){
		states_e[sct+s]+=states_e[s1]*states[si+s];
		states_e[si+s]+=states_e[s1]*states[sct+s];
		states_e[prev_cell_start+s]+=states_e[s1]*states[sf+s];
		states_e[sf+s]+=states_e[s1]*states[prev_cell_start+s];
		s1--;
	}
	
	//c + tanh
	for(s = c_state_size-1; s >= 0; s--){
		e = (1-states[s1]*states[s1])*states_e[s1];
		for(i = c_proj_size-1; i >= 0; i--){
			chars_e[char_start+i] += e*f_b_params[p];
			lstm_params_e[p--] += e*chars[char_start+i];
		}		
		for(i = c_state_size-1; i >= 0; i--){
			states_e[prev_state_start+i]+=e*f_b_params[p];
			lstm_params_e[p--] +=e*states[prev_state_start+i];
		}
		lstm_params_e[p--]+=e;
		s1--;		
	}
	
	
	//fgate
	for(s = c_state_size-1; s >= 0; s--){		
		e = states[s1]*(1-states[s1])*states_e[s1];
		for(i = c_proj_size-1; i >= 0; i--){
			chars_e[char_start+i] += e*f_b_params[p];
			lstm_params_e[p--] += e*chars[char_start+i];
		}
		for(i = c_cell_size-1; i >= 0; i--){
			states_e[prev_cell_start+i]+=e*f_b_params[p];
			lstm_params_e[p--] +=e*states[prev_cell_start+i];
		}
		for(i = c_state_size-1; i >= 0; i--){
			states_e[prev_state_start+i]+=e*f_b_params[p];
			lstm_params_e[p--] +=e*states[prev_state_start+i];
		}
		lstm_params_e[p--]+=e;
		s1--;
	}
	
	//igate
	for(s = c_state_size-1; s >= 0; s--){		
		e = states[s1]*(1-states[s1])*states_e[s1];
		for(i = c_proj_size-1; i >= 0; i--){
			chars_e[char_start+i] += e*f_b_params[p];
			lstm_params_e[p--] += e*chars[char_start+i];
		}
		for(i = c_cell_size-1; i >= 0; i--){
			states_e[prev_cell_start+i]+=e*f_b_params[p];
			lstm_params_e[p--] +=e*states[prev_cell_start+i];
		}
		for(i = c_state_size-1; i >= 0; i--){
			states_e[prev_state_start+i]+=e*f_b_params[p];
			lstm_params_e[p--] +=e*states[prev_state_start+i];
		}
		lstm_params_e[p--]+=e;
		s1--;
	}
	
	if(p+1!=pStart){
		printf("crap! p!= %d p = %d\n",pStart,p+1);
	}
	if(s1+1!=next_start){
		printf("crap! s1!= %d s1 = %d\n",next_start,s1+1);
	}
}

void lstmForward(char* word, int len, real* out, real *f_states, real *b_states, real *chars){
	//printf("%s\n",word);
	int i,s,c,p;
	for(s = 0; s < (len+1)*(c_state_size*7); s++){
		f_states[s]=0;
		b_states[s]=0;
	}
	for(s = 0; s < c_state_size; s++){
		f_states[c_state_size*3]=f_init_cell[s];
		f_states[c_state_size*6]=f_init_state[s];
		b_states[c_state_size*3]=b_init_cell[s];
		b_states[c_state_size*6]=b_init_state[s];
	}
	for(i = 0; i < len; i++){
		c = word[i];
		for(s = 0; s < c_proj_size; s++){
			chars[i*c_proj_size+s] = c_lookup[c*c_proj_size+s];
		}
	}
	
	for(i = 0; i < len; i++){
		lstmForwardBlock(chars, i*c_proj_size, f_states, (i+1)*c_state_size*7, 0);
	}
	for(i = 0; i < len; i++){
		lstmForwardBlock(chars, (len-i-1)*c_proj_size, b_states, (i+1)*c_state_size*7, c_lstm_params_number);
	}
	
	//printStates(f_states,c_state_size*7);

	for(s = 0; s < layer1_size; s++){
		out[s]=0;
	}	
	p=c_lstm_params_number*2;
	for(s = 0; s < layer1_size; s++){
		for(i = 0; i < c_state_size; i++){
			out[s]+=f_states[len*c_state_size*7+c_state_size*6 + i]*f_b_params[p++];			
			out[s]+=b_states[len*c_state_size*7+c_state_size*6 + i]*f_b_params[p++];			
		}
//		printf("%f ",out[s]);
	}
//	printf("\n");
}

void lstmBackward(char* word, int len, real* out, real *f_states, real *b_states, real* chars, real* out_e, real *f_states_e, real *b_states_e, real* chars_e, real *lstm_params_e){
	int i,s,c,p;
	for(s = 0; s < (len+1)*c_state_size*7; s++){
		f_states_e[s]=0;
		b_states_e[s]=0;
	}
	for(i = 0; i < len; i++){
		for(s = 0; s < c_proj_size; s++){
			chars_e[i*c_proj_size+s] = 0;
		}
	}
	for(i = 0; i < c_lstm_params_number*2; i++){
		lstm_params_e[i]=0;
	}
	
	p=c_lstm_params_number*2;
	for(s = 0; s < layer1_size; s++){
		for(i = 0; i < c_state_size; i++){
			f_states_e[len*c_state_size*7+c_state_size*6 + i]+=out_e[s]*f_b_params[p];
			f_b_params[p] += out_e[s] * f_states[len*c_state_size*7+c_state_size*6 + i];
			p++;
			b_states_e[len*c_state_size*7+c_state_size*6 + i]+=out_e[s]*f_b_params[p];
			f_b_params[p] += out_e[s] * b_states[len*c_state_size*7+c_state_size*6 + i];
			p++;
		}
	}
	
	for(i = len-1; i >=0; i--){
		lstmBackwardBlock(chars, i*c_proj_size, b_states, (i+1)*c_state_size*7, c_lstm_params_number, chars_e,b_states_e,lstm_params_e);
	}
	
	for(i = len-1; i >=0; i--){
		lstmBackwardBlock(chars, (len-i-1)*c_proj_size, f_states, (i+1)*c_state_size*7, 0, chars_e,f_states_e,lstm_params_e);
	}

	for(i = 0; i < len; i++){
		c = word[i];
		for(s = 0; s < c_proj_size; s++){
			c_lookup[c*c_proj_size+s] += chars_e[i*c_proj_size+s];
		}
	}
	
	for(s = 0; s < c_state_size; s++){
		f_init_cell[s]+=f_states_e[c_state_size*3];
		f_init_state[s]+=f_states_e[c_state_size*6];
		b_init_cell[s]+=b_states_e[c_state_size*3];
		b_init_state[s]+=b_states_e[c_state_size*6];
	}
	
	for(s = 0; s < c_lstm_params_number*2; s++){
		f_b_params[c]+=lstm_params_e[c];
	}
	
	//printf("out\n");
	//printStates(f_states,(len)*c_state_size*7);
	//printf("err\n");
	//printStates(f_states_e,(len)*c_state_size*7);

}

real hardTanh(real x){
	if(x>=1){
		return 1;
	}
	else if(x<=-1){
		return -1;
	}
	else{
		return x;
	}
}

real dHardTanh(real x, real g){
	if(x > 1 && g > 0){
		return 0;
	}
	if(x < -1 && g < 0){
		return 0;
	}
	return 1;
}

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
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

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Reads a word and returns its index in the vocabulary
int ReadAndStoreWordIndex(FILE *fin, char* word) {
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitClassUnigramTable() {
  long long a,c;
  printf("loading class unigrams \n");
  FILE *fin = fopen(negative_classes_file, "rb");
  if (fin == NULL) {
    printf("ERROR: class file not found!\n");
    exit(1);
  }
  word_to_group = (int *)malloc(vocab_size * sizeof(int));
  for(a = 0; a < vocab_size; a++) word_to_group[a] = -1;
  char class[MAX_STRING];
  char prev_class[MAX_STRING];
  prev_class[0] = 0;
  char word[MAX_STRING];
  class_number = -1;
  while (1) {
    if (feof(fin)) break;
    ReadWord(class, fin);
    ReadWord(word, fin);
    int word_index = SearchVocab(word);
    if (word_index != -1){
       if(strcmp(class, prev_class) != 0){
	    class_number++;
	    strcpy(prev_class, class);
       }
       word_to_group[word_index] = class_number;
    }
    ReadWord(word, fin);
  }
  class_number++;
  fclose(fin);
  
  group_to_table = (int *)malloc(table_size * class_number * sizeof(int)); 
  long long train_words_pow = 0;
  real d1, power = 0.75;
  
  for(c = 0; c < class_number; c++){
     long long offset = c * table_size;
     train_words_pow = 0;
     for (a = 0; a < vocab_size; a++) if(word_to_group[a] == c) train_words_pow += pow(vocab[a].cn, power);
     int i = 0;
     while(word_to_group[i]!=c && i < vocab_size) i++;
     d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
     for (a = 0; a < table_size; a++) {
	//printf("index %lld , word %d\n", a, i);
	group_to_table[offset + a] = i;
        if (a / (real)table_size > d1) {
	   i++;
           while(word_to_group[i]!=c && i < vocab_size) i++;
	   d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
        }
        if (i >= vocab_size) while(word_to_group[i]!=c && i >= 0) i--;
     }
  }
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  window_layer_size = layer1_size*window*2;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&syn1_window, 128, (long long)vocab_size * window_layer_size * sizeof(real));
    if (syn1_window == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&syn_hidden_word, 128, (long long)vocab_size * window_hidden_size * sizeof(real));
    if (syn_hidden_word == NULL) {printf("Memory allocation failed\n"); exit(1);}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
    for (a = 0; a < vocab_size; a++) for (b = 0; b < window_layer_size; b++)
     syn1_window[a * window_layer_size + b] = 0;
    for (a = 0; a < vocab_size; a++) for (b = 0; b < window_hidden_size; b++)
     syn_hidden_word[a * window_hidden_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&syn1neg_window, 128, (long long)vocab_size * window_layer_size * sizeof(real));
    if (syn1neg_window == NULL) {printf("Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&syn_hidden_word_neg, 128, (long long)vocab_size * window_hidden_size * sizeof(real));
    if (syn_hidden_word_neg == NULL) {printf("Memory allocation failed\n"); exit(1);}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
    for (a = 0; a < vocab_size; a++) for (b = 0; b < window_layer_size; b++)
     syn1neg_window[a * window_layer_size + b] = 0;
    for (a = 0; a < vocab_size; a++) for (b = 0; b < window_hidden_size; b++)
     syn_hidden_word_neg[a * window_hidden_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }

  a = posix_memalign((void **)&syn_window_hidden, 128, window_hidden_size * window_layer_size * sizeof(real));
  if (syn_window_hidden == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < window_hidden_size * window_layer_size; a++){
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn_window_hidden[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (window_hidden_size*window_layer_size);
  }
  
  a = posix_memalign((void **)&c_lookup, 128, (long long)C_MAX_CODE * c_proj_size * sizeof(real));
  if (c_lookup == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < C_MAX_CODE * c_proj_size; a++){
    next_random = next_random * (unsigned long long)25214903917 + 11;
    c_lookup[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (c_proj_size);
  }

  a = posix_memalign((void **)&f_init_state, 128, c_state_size * sizeof(real));
  if (f_init_state == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&f_init_cell, 128, c_state_size * sizeof(real));
  if (f_init_cell == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&b_init_state, 128, c_state_size * sizeof(real));
  if (b_init_state == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&b_init_cell, 128, c_state_size * sizeof(real));
  if (b_init_cell == NULL) {printf("Memory allocation failed\n"); exit(1);}

  for (a = 0; a < c_state_size; a++){
    next_random = next_random * (unsigned long long)25214903917 + 11;
    f_init_state[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (c_state_size);
    next_random = next_random * (unsigned long long)25214903917 + 11;
    f_init_cell[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (c_state_size);
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b_init_state[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (c_state_size);
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b_init_cell[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (c_state_size);
  }

  c_lstm_params_number = /*input*/ (c_state_size+c_cell_size+c_proj_size+1)*c_state_size +
  /*forget*/ (c_state_size+c_cell_size+c_proj_size+1)*c_state_size +
  /*cell*/ (c_state_size+c_proj_size+1)*c_state_size +
  /*output*/ (c_state_size+c_cell_size+c_proj_size+1)*c_state_size;
  
  c_params_number = ( c_lstm_params_number * 2 + (c_state_size*2)*layer1_size) ;
  a = posix_memalign((void **)&f_b_params, 128, c_params_number* sizeof(real));
  if (f_b_params == NULL) {printf("Memory allocation failed\n"); exit(1);}

  for (a = 0; a < c_params_number; a++){
    next_random = next_random * (unsigned long long)25214903917 + 11;
    f_b_params[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / (c_state_size+c_cell_size+c_proj_size);
  }

  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  char c_sen[(MAX_SENTENCE_LENGTH + 1) * MAX_STRING];
  unsigned long long next_random = (long long)id;
  real f, g, acc_g=0;
  clock_t now;
  int input_len_1 = layer1_size;
  if(type == 2 || type == 4){
     input_len_1=window_layer_size;
  }
  real *neu1 = (real *)calloc(input_len_1, sizeof(real));
  real *neu1e = (real *)calloc(input_len_1, sizeof(real));

  int input_len_2 = 0;
  if(type == 4){
     input_len_2 = window_hidden_size;
  }
  real *neu2 = (real *)calloc(input_len_2, sizeof(real));
  real *neu2e = (real *)calloc(input_len_2, sizeof(real));

  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  
  real *f_states = (real *)calloc((c_state_size * 7) * (MAX_STRING + 1), sizeof(real));
  real *f_states_e = (real *)calloc((c_state_size * 7) * (MAX_STRING + 1), sizeof(real));
  real *b_states = (real *)calloc((c_state_size * 7) * (MAX_STRING + 1), sizeof(real));
  real *b_states_e = (real *)calloc((c_state_size * 7) * (MAX_STRING + 1), sizeof(real));
  real *chars = (real *)calloc(c_proj_size * MAX_STRING, sizeof(real));
  real *chars_e = (real *)calloc(c_proj_size * MAX_STRING, sizeof(real));
  real *lstm_params_e = (real *)calloc(c_lstm_params_number*2, sizeof(real));
  
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk : error %.4f", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000), acc_g);
         acc_g=0;
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadAndStoreWordIndex(fi, &c_sen[sentence_length*MAX_STRING]);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < input_len_1; c++) neu1[c] = 0;
    for (c = 0; c < input_len_1; c++) neu1e[c] = 0;
    for (c = 0; c < input_len_2; c++) neu2[c] = 0;
    for (c = 0; c < input_len_2; c++) neu2e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (type == 0) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
	    if(word_to_group != NULL && word_to_group[word] != -1){
		target = word;
		while(target == word) {
			target = group_to_table[word_to_group[word]*table_size + (next_random >> 16) % table_size];
            		next_random = next_random * (unsigned long long)25214903917 + 11;
		}
		//printf("negative sampling %lld for word %s returned %s\n", d, vocab[word].word, vocab[target].word);
	    }
	    else{
            	target = table[(next_random >> 16) % table_size];
	    }
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } else if(type==1) {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
	    next_random = next_random * (unsigned long long)25214903917 + 11;
            if(word_to_group != NULL && word_to_group[word] != -1){
                target = word;
                while(target == word) {
                        target = group_to_table[word_to_group[word]*table_size + (next_random >> 16) % table_size];
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                }
                //printf("negative sampling %lld for word %s returned %s\n", d, vocab[word].word, vocab[target].word);
            }
            else{
                target = table[(next_random >> 16) % table_size];
            }
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    else if(type == 2){ //train the cwindow architecture
      // in -> hidden
      cw = 0;
      for (a = 0; a < window * 2 + 1; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        window_offset = a*layer1_size;
        if (a > window) window_offset-=layer1_size;
        for (c = 0; c < layer1_size; c++) neu1[c+window_offset] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * window_layer_size;
          // Propagate hidden -> output
          for (c = 0; c < window_layer_size; c++) f += neu1[c] * syn1_window[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < window_layer_size; c++) neu1e[c] += g * syn1_window[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < window_layer_size; c++) syn1_window[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            if(word_to_group != NULL && word_to_group[word] != -1){
                target = word;
                while(target == word) {
                        target = group_to_table[word_to_group[word]*table_size + (next_random >> 16) % table_size];
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                }
                //printf("negative sampling %lld for word %s returned %s\n", d, vocab[word].word, vocab[target].word);
            }
            else{
                target = table[(next_random >> 16) % table_size];
            }
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * window_layer_size;
          f = 0;
          for (c = 0; c < window_layer_size; c++) f += neu1[c] * syn1neg_window[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          acc_g+=g;
          for (c = 0; c < window_layer_size; c++) neu1e[c] += g * syn1neg_window[c + l2];
          for (c = 0; c < window_layer_size; c++) syn1neg_window[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = 0; a < window * 2 + 1; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
	  window_offset = a * layer1_size;
	  if(a > window) window_offset -= layer1_size;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c + window_offset];
        }
      }
    }
    else if (type == 3){  //train structured skip-gram
      for (a = 0; a < window * 2 + 1; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        char* c_last_word = &c_sen[c*MAX_STRING];
        if(rep == 1){
        	lstmForward(c_last_word, strlen(c_last_word),neu1, f_states, b_states, chars);
        }        
        else{
        	for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + l1];
        }
	window_offset = a * layer1_size;
	if(a > window) window_offset -= layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * window_layer_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1_window[c + l2 + window_offset];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1_window[c + l2 + window_offset];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2 + window_offset] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
	     next_random = next_random * (unsigned long long)25214903917 + 11;
            if(word_to_group != NULL && word_to_group[word] != -1){
                target = word;
                while(target == word) {
                        target = group_to_table[word_to_group[word]*table_size + (next_random >> 16) % table_size];
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                }
                //printf("negative sampling %lld for word %s returned %s\n", d, vocab[word].word, vocab[target].word);
            }
            else{
                target = table[(next_random >> 16) % table_size];
            }
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * window_layer_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg_window[c + l2 + window_offset];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          acc_g+=g;

          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_window[c + l2 + window_offset];
          for (c = 0; c < layer1_size; c++) syn1neg_window[c + l2 + window_offset] += g * neu1[c];
        }
        // Learn weights input -> hidden
        
        if(rep == 1){
        	lstmBackward(c_last_word, strlen(c_last_word),neu1, f_states, b_states, chars, neu1e,f_states_e, b_states_e, chars_e, lstm_params_e);
        }
        else{
        	for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
      }
    }
    else if(type == 4){ //training senna
	// in -> hidden
      cw = 0;
      for (a = 0; a < window * 2 + 1; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        window_offset = a*layer1_size;
        if (a > window) window_offset-=layer1_size;
        for (c = 0; c < layer1_size; c++) neu1[c+window_offset] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
		for (a = 0; a < window_hidden_size; a++){
          c = a*window_layer_size;
          for(b = 0; b < window_layer_size; b++){
             neu2[a] += syn_window_hidden[c + b] * neu1[b];
          }
        }
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * window_hidden_size;
          // Propagate hidden -> output
          for (c = 0; c < window_hidden_size; c++) f += hardTanh(neu2[c]) * syn_hidden_word[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < window_hidden_size; c++) neu2e[c] += dHardTanh(neu2[c],g) * g * syn_hidden_word[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < window_hidden_size; c++) syn_hidden_word[c + l2] += dHardTanh(neu2[c],g) * g * neu2[c];
        }
      // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
	    next_random = next_random * (unsigned long long)25214903917 + 11;
            if(word_to_group != NULL && word_to_group[word] != -1){
                target = word;
                while(target == word) {
                        target = group_to_table[word_to_group[word]*table_size + (next_random >> 16) % table_size];
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                }
                //printf("negative sampling %lld for word %s returned %s\n", d, vocab[word].word, vocab[target].word);
            }
            else{
                target = table[(next_random >> 16) % table_size];
            }
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * window_hidden_size;
          f = 0;
          for (c = 0; c < window_hidden_size; c++) f += hardTanh(neu2[c]) * syn_hidden_word_neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha / negative;
          else if (f < -MAX_EXP) g = (label - 0) * alpha / negative;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha / negative;
          for (c = 0; c < window_hidden_size; c++) neu2e[c] += dHardTanh(neu2[c],g) * g * syn_hidden_word_neg[c + l2];
          for (c = 0; c < window_hidden_size; c++) syn_hidden_word_neg[c + l2] += dHardTanh(neu2[c],g) * g * neu2[c];
        }
        for (a = 0; a < window_hidden_size; a++)
          for(b = 0; b < window_layer_size; b++)
	     neu1e[b] += neu2e[a] * syn_window_hidden[a*window_layer_size + b];
	for (a = 0; a < window_hidden_size; a++)
          for(b = 0; b < window_layer_size; b++)
	     syn_window_hidden[a*window_layer_size + b] += neu2e[a] * neu1[b];
        // hidden -> in
        for (a = 0; a < window * 2 + 1; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          window_offset = a * layer1_size;
          if(a > window) window_offset -= layer1_size;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c + window_offset];
        }
      }
    }
    else{
	printf("unknown type %i", type);
	exit(0);
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  if (negative_classes_file[0] != 0) InitClassUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
	real *f_states = (real *)calloc((c_state_size * 7) * (MAX_STRING + 1), sizeof(real));
	real *b_states = (real *)calloc((c_state_size * 7) * (MAX_STRING + 1), sizeof(real));
  	real *chars = (real *)calloc(c_proj_size * MAX_STRING, sizeof(real));
	real *neu1 = (real *)calloc(layer1_size * MAX_STRING, sizeof(real));

    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if(rep==1){
      	lstmForward(vocab[a].word, strlen(vocab[a].word),neu1, f_states,b_states,chars);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&neu1[b], sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", neu1[b]);
      }
      else{
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t-negative-classes <file>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-type <int>\n");
    printf("\t\tType of embeddings (0 for cbow, 1 for skipngram, 2 for cwindow, 3 for structured skipngram, 4 for senna type)\n");
    printf("\t-rep <int>\n");
    printf("\t\tType of word rep (0 for word, 1 for character\n");
    printf("\t-char-state-dim <int>\n");
    printf("\t\tcharacter state size\n");
    printf("\t-char-proj-dim <int>\n");
    printf("\t\tcharacter projection size\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -type 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  negative_classes_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-type", argc, argv)) > 0) type = atoi(argv[i + 1]);
  if (type==0 || type==2 || type==4) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative-classes", argc, argv)) > 0) strcpy(negative_classes_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-rep", argc, argv)) > 0) rep = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-char-state-dim", argc, argv)) > 0) {c_state_size = atoi(argv[i + 1]); c_cell_size = c_state_size;}
  if ((i = ArgPos((char *)"-char-proj-dim", argc, argv)) > 0) {c_proj_size = atoi(argv[i + 1]);}
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  tanhTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    tanhTable[i] = tanh((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
  }
  TrainModel();
  return 0;
}
