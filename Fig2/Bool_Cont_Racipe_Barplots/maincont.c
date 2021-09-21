#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include "pcg_basic.h"
#include <omp.h>
#include <math.h>

typedef FILE *fileptr;

double max(double a, double b){
    if(a > b){
        return a;
    }
    else{
        return b;
    }
}

double min(double a, double b){
    if(a < b){
        return a;
    }
    else{
        return b;
    }
}

int main(){
    double begin1 = omp_get_wtime();

    int updation_rule = 2;
    int nsim = 1000;
    int maxtime = 2000;

    int numnodes;   //automatically set later
    int numruns;
    int constantnode;
    char filename[5000];
    char input[5000];
    char output[5000];
    int ac = 10;

    double *link_matrix;
    char **nodes;


    /* READING FROM init.txt */
    fileptr init;
    // fileptr jsdfile;
    char temp1[300], temp2[300], temp3[300];
    int tempint;

    init = fopen("init.txt", "r");
    if(init == NULL){
        fprintf(stderr, "ERROR: Can't open init.txt!\n");
        exit(0);
    }
    fscanf(init, "%s %s\n", temp1, input);
    fscanf(init, "%s %s\n", temp1, output);
    fscanf(init, "%s %s\n", temp1, filename);
    fscanf(init, "%s %d\n", temp1, &numruns);
    fscanf(init, "%s %d\n", temp1, &nsim);
    fscanf(init, "%s %d\n", temp1, &maxtime);
    fscanf(init, "%s %d\n", temp1, &constantnode);


    /* setting topo and ids file name variables */
    char topofilename[5000];
    strcpy(topofilename, "input/");
    strcat(topofilename, filename);
    strcat(topofilename, ".topo");

    char idsfilename[5000];
    strcpy(idsfilename, "input/");
    strcat(idsfilename, filename);
    strcat(idsfilename, ".ids");


    /* PARSING IDS FILE */
    fileptr idsfile = fopen(idsfilename, "r");

    if(idsfile == NULL){
        fprintf(stderr, "ERROR: ids file cannot be opened, or is missing.\n");
        exit(0);
    }

    char temp4[300], temp5[300];
    int count = 0;

    fscanf(idsfile, "%s %s\n", temp4, temp5);

    while(fscanf(idsfile, "%s %s\n", temp4, temp5) != EOF){
        sscanf(temp5, "%d", &tempint);
        count++;
    }

    numnodes = count;

    fclose(idsfile);
    idsfile = fopen(idsfilename, "r");
    nodes = (char **) malloc((numnodes) * sizeof(char *));
    for(int i = 0; i < numnodes; i++){
        nodes[i] = (char *) malloc(50 * sizeof(char));
    }

    fscanf(idsfile, "%s %s\n", temp1, temp2);
    while(fscanf(idsfile, "%s %s\n", temp1, temp2) != EOF){
        sscanf(temp2, "%d", &tempint);
        strcpy(nodes[tempint], temp1);
    }

    /* PARSING TOPO FILE */
    fileptr topofile;
    topofile = fopen(topofilename, "r");
    if(topofile == NULL){
        fprintf(stderr, "ERROR: topo file cannot be opened, or is missing.\n");
        exit(0);
    }

    int n = numnodes;
    link_matrix = (double *) malloc(n * n * sizeof(double));
    if(link_matrix == NULL){
        fprintf(stderr, "ERROR: link matrix couldn't be initialised (malloc error).\n");
        exit(0);
    }

    for(int i = 0; i < n * n; i++){
        link_matrix[i] = 0;
    }
    int index1 = 0, index2 = 0;

    fscanf(topofile, "%s %s %s\n", temp1, temp2, temp3);

    while(fscanf(topofile, "%s %s %s\n", temp1, temp2, temp3) != EOF){
        sscanf(temp3, "%d", &tempint);
        for(int i = 0; i < n; i++){
            if(strcmp(nodes[i], temp1) == 0){
                index1 = i;
            }
            if(strcmp(nodes[i], temp2) == 0){
                index2 = i;
            }
        }
        link_matrix[index1 * n + index2] = tempint == 1 ? 1 : -1;
    }

    pcg32_random_t rng;
    double time1 = (omp_get_wtime() - begin1) * 100000;
    pcg32_srandom_r(&rng, time(NULL) ^ (intptr_t) & printf, (intptr_t) & time1);

    for(int pp = 0; pp < numruns; pp++){
        int cnt = 0;
        int level = pp / (2 * ac) + 1;
        int saveq = 0;
        int savesign = 0;

        double deg[numnodes];

        for(int q = 0; q < numnodes; q++){

            deg[q] = 0;
        }
        for(int q = 0; q < numnodes; q++){
            for(int m = 0; m < numnodes; m++){
                deg[q] += fabs(link_matrix[m * numnodes + q]);
            }
        }

        for(int q = 0; q < numnodes; q++){
            deg[q] = 0.25 * pow(deg[q], 0.5);
        }

        char initname[100], ssname[100], nssname[100]; //fssname[100]
        strcpy(initname, "output/");
        strcpy(ssname, "output/");
        strcpy(nssname, "output/");

        char snum[100];
        sprintf(snum, "%d", pp + 1);

        strcat(initname, filename);
        strcat(initname, "_init_run");
        strcat(initname, snum);
        strcat(initname, ".txt");

        strcat(ssname, filename);
        strcat(ssname, "_ss_run");
        strcat(ssname, snum);
        strcat(ssname, ".txt");

        strcat(nssname, filename);
        strcat(nssname, "_nss_run");
        strcat(nssname, snum);
        strcat(nssname, ".txt");

        fileptr initstate, ss, nss; //fss
        initstate = fopen(initname, "w");

        ss = fopen(ssname, "w");

        nss = fopen(nssname, "w");



        /* Initial file inputs */

        fprintf(initstate, "ID ");

        fprintf(ss, "ID ");
        fprintf(nss, "ID ");

        for(int i = 0; i < numnodes - 1; i++){
            fprintf(initstate, "%s ", nodes[i]);
            fprintf(ss, "%s ", nodes[i]);
            fprintf(nss, "%s ", nodes[i]);
        }
        for(int i = numnodes - 1; i < numnodes; i++){
            fprintf(initstate, "%s", nodes[i]);
            fprintf(ss, "%s", nodes[i]);
            fprintf(nss, "%s", nodes[i]);
        }
        fprintf(initstate, " Stable");
        fprintf(initstate, "\n");
        fprintf(ss, "\n");
        fprintf(nss, "\n");

        int simnum = 0;
        int idnum = 1;


        while(simnum < nsim){
            fprintf(initstate, "%d ", idnum);

            int stable_checker = 1;
            double activation[numnodes];
            double curstate[numnodes];

            //randomise curstate
            for(int i = 0; i < numnodes; i++){
                curstate[i] = ldexp(pcg32_random_r(&rng), -32) * 2 - 1;
                fprintf(initstate, "%d ", ((int) (curstate[i] > 0.5)) * 2 - 1);
            }

            //matmul and stable state checker
            for(int iter = 0; iter < maxtime; iter++){
                stable_checker = 1;
                for(int i = 0; i < numnodes; i++){
                    activation[i] = 0;
                    for(int j = 0; j < numnodes; j++){
                        activation[i] += link_matrix[j * numnodes + i] * curstate[j];
                    }
                    if(activation[i] * curstate[i] < 0 && fabs(activation[i]) >= deg[i]){ //
                        stable_checker = 0;
                    }

                }

                if(stable_checker){
                    fprintf(ss, "%d ", idnum);

                    for(int i = 0; i < numnodes - 1; i++){
                        fprintf(ss, "%d ", ((int) (curstate[i] > 0) * 2) - 1);
                    }
                    fprintf(ss, "%d", (int) (curstate[numnodes - 1] > 0) * 2 - 1);

                    fprintf(ss, "\n");

                    fprintf(initstate, "1\n");
                    idnum++;
                    simnum++;
                    break;
                }
                else{
                    int rand_node;
                    int qq = 0;
                    switch(updation_rule){
                        case 0: //sync updation
                            for(int i = 0; i < numnodes; i++){
                                if(activation[i] > 0){
                                    curstate[i] = 1;
                                }
                                else if(activation[i] < 0){
                                    curstate[i] = -1;
                                }
                            }
                            break;
                        case 1: //async updation

                            while(qq == 0){
                                rand_node = pcg32_boundedrand_r(&rng, numnodes);
                                if(activation[rand_node] > 0 && curstate[rand_node] < 0){
                                    curstate[rand_node] = 1;
                                    qq = 1;
                                }
                                else if(activation[rand_node] < 0 && curstate[rand_node] > 0){
                                    curstate[rand_node] = -1;
                                    qq = 1;
                                }
                            }
                            break;
                        case 2: //sync updation continuous
                            for(int i = 0; i < numnodes; i++){
                                if(fabs(activation[i]) > 0){

                                    curstate[i] += 0.1 * activation[i] / (fabs(activation[i]) + 1);
                                }

                                if(curstate[i] > 1){
                                    curstate[i] = 1;
                                }

                                if(curstate[i] < -1){
                                    curstate[i] = -1;
                                }



                            }
                            break;
                        case 3: //async updation continuous
                            while(qq == 0){

                                rand_node = 1 + pcg32_boundedrand_r(&rng, numnodes - 1);
                                if(iter > maxtime / 1.1){
                                    printf("%d %d\n", iter, rand_node);
                                }

                                if(activation[rand_node] > 0 && fabs(curstate[rand_node]) < 1){
                                    qq = 1;
                                }
                                if(qq == 1){
                                    if(fabs(activation[rand_node]) >= -1){
                                        if(activation[rand_node] > 0){
                                            curstate[rand_node] += 0.1;
                                        }
                                        if(activation[rand_node] < 0){
                                            curstate[rand_node] += -0.1;
                                        }
                                    }
                                    if(curstate[rand_node] > 1){
                                        curstate[rand_node] = 1;
                                    }
                                    else if(curstate[rand_node] < -1){

                                        curstate[rand_node] = -1;
                                    }
                                    break;
                                }
                            }
                            break;
                    }
                }
            }

            if(stable_checker == 0){
                //printf("sdsds \n");
                fprintf(ss, "%d ", idnum);
                for(int i = 0; i < numnodes - 1; i++){
                    fprintf(ss, "%d ", (int) (curstate[i] > 0) * 2 - 1);
                }
                fprintf(ss, "%d", (int) (curstate[numnodes - 1] > 0) * 2 - 1);
                fprintf(ss, "\n");
                fprintf(initstate, "0\n");
                idnum++;
                simnum++;
            }
            if(simnum % (nsim / 100) == 0){
                printf("\r %s %d %d%% %f s ", filename, pp + 1, ((simnum * 100) / nsim), omp_get_wtime() - begin1);
                fflush(stdout);
            }
        }
        link_matrix[saveq] = savesign;

        fclose(initstate);

        fclose(ss);
        fclose(nss);
    }

    fclose(init);
    fclose(idsfile);
    fclose(topofile);

    free(link_matrix);
    free(nodes);
    return 0;
}