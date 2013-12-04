#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void main(int argc, char *argv[]){

  char *header;
  int i, ntri, j, k, failed;
  float dr, eps, dist, mind, maxd;
  float n[3], s[3];
  float v[3][3];
  short dum;
  FILE *f;

  if(argc!=3){
    printf("Wrong input arguments.\nCorrect usage: ./test-triangle-size filename.stl dr\nExample: ./test-triangle-size box.stl 0.1\nEXIT\n");
    return;
  }

  dr = atof(argv[2]);
  eps = dr*1e-4;
  failed = 0;

  f = fopen(argv[1], "rb");
  header = (char*) malloc(80*sizeof(char));
  fread(header, 80*sizeof(char), 1, f);
  fread(&ntri, sizeof(int), 1, f);
  printf("Discretization dr: %f\n", dr);
  printf("Reading %d triangles from file %s.\n\n", ntri, argv[1]);
  mind = 1e10;
  maxd = 0e10;
  for(i=0; i<ntri; i++){
    fread(n,sizeof(float),3,f);
    for(j=0; j<3; j++){
      fread(v[j],sizeof(float),3,f);
      s[j] = 0.0;
    }
    fread(&dum, sizeof(short), 1, f);
    for(j=0; j<3; j++){
      for(k=0; k<3; k++)
        s[k] += v[j][k]/3.0;
    }
    for(j=0; j<3; j++){
      dist = sqrt(pow(s[0]-v[j][0],2)+pow(s[1]-v[j][1],2)+pow(s[2]-v[j][2],2.0));
      if(dist>maxd)
        maxd = dist;
      if(dist<mind)
        mind = dist;
      if(dist>dr-eps){
        printf("\tFailed: %d, %d, %f, %f\n", i, j, dist, dr-eps);
        failed++;
      }
    }
  }
  fclose(f);
  printf("\nMinimum distance: %f\nMaximum distance: %f\n", mind, maxd);
  if(failed==0)
    printf("All triangles passed the test.\n");
  else
    printf("%d triangles failed the test.\n",failed);
  return;
}
