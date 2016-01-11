/*
    A Bayesian partition method to estimate co-modules underlying two matrices
    Copyright (C) 2012 Shiwen Zhao
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/timeb.h>

#include"header.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_exp.h>


using namespace std;

int numD;
int numG;
int numP;
double prior_C = 2.0;
#define Scale 1

double Delta_Beta = 0.2;

Hyperparameters hyperpars;

struct_Module_Count Module_Count;
struct_SS SS;
struct_Sum_Num Sum_Num;
struct_Posterior Posterior;


void Normalize_Matrix(gsl_matrix* Matrix, gsl_matrix* Matrix_new)
{
    int x = Matrix->size1;
    int y = Matrix->size2;
    double mean,std;
    double tempt;

    for(int j = 0; j<y; j++)
    {
        mean = gsl_stats_mean(&(Matrix->data[j]),Matrix->tda,x);
        std = gsl_stats_sd(&(Matrix->data[j]),Matrix->tda,x);
        if(std == 0)
        {
            cout<<"Error: The "<<j<<"th column has a zero standard deviation"<<endl;
            exit(1);
        }
        for(int i = 0; i<x; i++)
        {
            tempt = Matrix->data[i*y + j];
            Matrix_new->data[i*y + j] = (tempt-mean)/std;
        }
    }
}

void import_Initial_Interation_Point(struct_Indicator& Indicator, struct_Module_Center& Module_Center,\
                                     char *Initial_Ig_file, char *Initial_Imd_file, char *Initial_Imp_file, \
                                     char *Initial_Beta_mid_file, char* Initial_Beta_mip_file)
{
    int M = Module_Count.M;
    int i,mm;
    int group;
    double d_value;


    //============================================== read initial point of Ig
    ifstream readData(Initial_Ig_file);
    if(readData == NULL)
    {
        cout<<"Error: There is a problem in opening "<<Initial_Ig_file<<endl;
        exit(1);
    }
    if(Indicator.Ig == NULL)
        Indicator.Ig = (int*)malloc(sizeof(int)*numG);
    else
    {
        free(Indicator.Ig);
        Indicator.Ig = (int*)malloc(sizeof(int)*numG);
    }
    i = 0;
    while(readData >> group)
        Indicator.Ig[i++] = group;
    readData.close();


    //============================================== read initial point of Imd
    readData.open(Initial_Imd_file);
    if(readData == NULL)
    {
        cout<<"Error: There is a problem in opening "<<Initial_Imd_file<<endl;
        exit(1);
    }
    if(Indicator.Imd == NULL)
        Indicator.Imd = (int*)malloc(sizeof(int)*numD*(M+1));
    else
    {
        free(Indicator.Imd);
        Indicator.Imd = (int*)malloc(sizeof(int)*numD*(M+1));
    }

    for(mm = 0; mm <= M; mm++)
    {
        for(i = 0; i<numD; i++)
        {
            readData >> group;
            Indicator.Imd[i*(M+1) + mm] = group;
        }
    }
    readData.close();


    //============================================== read initial point of Imp
    readData.open(Initial_Imp_file);
    if(readData == NULL)
    {
        cout<<"Error: There is a problem in opening "<<Initial_Imp_file<<endl;
        exit(1);
    }
    if(Indicator.Imp == NULL)
    {
        Indicator.Imp = (int*)malloc(sizeof(int)*numP*(M+1));
    }
    else
    {
        free(Indicator.Imp);
        Indicator.Imp = (int*)malloc(sizeof(int)*numP*(M+1));
    }
    for(mm = 0; mm <= M; mm++)
    {
        for(i = 0; i<numP; i++)
        {
            readData >> group;
            Indicator.Imp[i*(M+1) + mm] = group;
        }
    }
    readData.close();


    //========================================================= read initial point of beta_mid
    readData.open(Initial_Beta_mid_file);
    if(readData == NULL)
    {
        cout<<"Error: There is a problem in opening "<<Initial_Beta_mid_file<<endl;
        exit(1);
    }
    if(Module_Center.Beta_mid == NULL)
        Module_Center.Beta_mid = (double*)malloc(sizeof(double)*numD*(M+1));
    else
    {
        free(Module_Center.Beta_mid);
        Module_Center.Beta_mid = (double*)malloc(sizeof(double)*numD*(M+1));
    }
    for(mm = 0; mm <= M; mm++)
    {
        for(i = 0; i<numD; i++)
        {
            readData >> d_value;
            Module_Center.Beta_mid[i*(M+1) + mm] = d_value;
        }
    }
    readData.close();


    //========================================================= read initial point of beta_mip
    readData.open(Initial_Beta_mip_file);
    if(readData == NULL)
    {
        cout<<"Error: There is a problem in opening "<<Initial_Beta_mip_file<<endl;
        exit(1);
    }

    if(Module_Center.Beta_mip == NULL)
        Module_Center.Beta_mip = (double*)malloc(sizeof(double)*numP*(M+1));
    else
    {
        free(Module_Center.Beta_mip);
        Module_Center.Beta_mip = (double*)malloc(sizeof(double)*numP*(M+1));
    }
    for(mm = 0; mm <= M; mm++)
    {
        for(i = 0; i<numP; i++)
        {
            readData >> d_value;
            Module_Center.Beta_mip[i*(M+1) + mm] = d_value;
        }
    }
    readData.close();
}


void read_matrix(string str,gsl_matrix* Matrix)
{
    FILE *inputfile;
    const char *array = str.c_str();
    inputfile = fopen(array,"r");
    if(inputfile == NULL)
    {
        cout<<"Error: There is a problem in opening "<<str<<endl;
        exit(1);
    }
    int status = gsl_matrix_fscanf(inputfile, Matrix);
    if(status == GSL_EFAILED)
    {
        cout<<"Error: The dimension in "<<str<<" may not equal the dimension inserted."<<endl;
        exit(1);
    }
    fclose(inputfile);

}

void initialize_hyperparameters(gsl_matrix* Matrix1,gsl_matrix* Matrix2)
{
    hyperpars.C_D = prior_C;
    hyperpars.C_G = prior_C;
    hyperpars.C_M = prior_C;
    hyperpars.C_P = prior_C;

    hyperpars.k_0d = 1.0;
    hyperpars.v_0d = 1.0;
    hyperpars.S2_0d = gsl_stats_variance(Matrix1->data,1,numD*numG);

    hyperpars.k_0p = 1.0;
    hyperpars.v_0p = 1.0;
    hyperpars.S2_0p = gsl_stats_variance(Matrix2->data,1,numP*numG);

    hyperpars.k_taod = 1.0;
    hyperpars.v_taod = 1.0;
    hyperpars.S2_taod = hyperpars.S2_0d/2;

    hyperpars.k_taop = 1.0;
    hyperpars.v_taop = 1.0;
    hyperpars.S2_taop = hyperpars.S2_0p/2;

    hyperpars.k_sigmad = 1.0/10;
    hyperpars.v_sigmad = numD*1.0/3;
    hyperpars.S2_sigmad = hyperpars.S2_0d/100;

    hyperpars.k_sigmap = 1.0/10;
    hyperpars.v_sigmap = numP*1.0/3;
    hyperpars.S2_sigmap = hyperpars.S2_0p/100;
}

void initialize_Indicator(struct_Indicator& Indicator,gsl_rng* r)
{
    //gsl_rng * r;
    //timeb this_time;
    //ftime(&this_time);
    //gsl_rng_default_seed = (unsigned long)(this_time.millitm)*getpid();
    //r = gsl_rng_alloc(gsl_rng_ranlxd1);
    //------------------------------------
    int i;
    int M = Module_Count.M;
    int *group;
    group = (int*)malloc(sizeof(int)*(M+1));
    if(Indicator.Ig == NULL)
        Indicator.Ig = (int*)malloc(sizeof(int)*numG);
    for(i = 0; i<=M; i++)
    {
        *(group+i) = i;
    }
    gsl_ran_sample(r,Indicator.Ig,numG,group,M+1,sizeof(int));
    //-----------------------------------
    if(Indicator.Imd == NULL)
        Indicator.Imd = (int*)malloc(sizeof(int)*numD*(M+1));
    if(Indicator.Imp == NULL)
        Indicator.Imp = (int*)malloc(sizeof(int)*numP*(M+1));
    gsl_ran_sample(r,Indicator.Imd,numD*(M+1),group,2,sizeof(int));
    gsl_ran_sample(r,Indicator.Imp,numP*(M+1),group,2,sizeof(int));
    free(group);
    //gsl_rng_free(r);
}

void initialize_Module_Center0(struct_Module_Center& Module_Center)
{
    int M = Module_Count.M;
    int i;
    if(Module_Center.Beta_mid == NULL)
    {
        Module_Center.Beta_mid = (double*)malloc(sizeof(double)*numD*(M+1));
        Module_Center.Beta_mip = (double*)malloc(sizeof(double)*numP*(M+1));
    }
    for (i = 0; i<numD*(M+1); i++)
        Module_Center.Beta_mid[i] = 0.0;
    for (i = 0; i<numP*(M+1); i++)
        Module_Center.Beta_mip[i] = 0.0;
}

void compute_SS(gsl_matrix* Matrix1, gsl_matrix* Matrix2,const struct_Indicator& Indicator, const struct_Module_Center& Module_Center)
{
    int M = Module_Count.M;
    int mm;
    int i,j;
    if(SS.SS_mbd == NULL)
    {
        SS.SS_md = (double*)malloc(sizeof(double)*(M+1));
        SS.SS_mp = (double*)malloc(sizeof(double)*(M+1));
        SS.SS_mbd = (double*)malloc(sizeof(double)*(M+1));
        SS.SS_mbp = (double*)malloc(sizeof(double)*(M+1));
    }

    double sum_Beta_mid[M+1], sum_Beta2_mid[M+1];
    double sum_Beta_mip[M+1], sum_Beta2_mip[M+1];
    int num_D[M+1],num_P[M+1];
    for (mm = 0; mm<=M;mm++)
    {
        SS.SS_md[mm] = 0.0;
        SS.SS_mp[mm] = 0.0;
        SS.SS_mbd[mm] = 0.0;
        SS.SS_mbp[mm] = 0.0;
    }

    double sum_Data;
    double tempt;

    for (j = 0; j<numG;j++)
    {
        mm = Indicator.Ig[j];
        sum_Data = 0.0;
        sum_Beta_mid[mm] = 0.0;
        sum_Beta2_mid[mm] = 0.0;
        num_D[mm] = 0;
        for(i = 0; i<numD;i++)
        {
            if(Indicator.Imd[i*(M+1)+mm] == 0)
            {
                tempt = Matrix1->data[i*numG + j];
                SS.SS_md[mm] += tempt * tempt;
            }
            else
            {
                tempt = Matrix1->data[i*numG + j] - Module_Center.Beta_mid[i*(M+1)+mm];
                SS.SS_md[mm] += tempt * tempt;
                sum_Beta_mid[mm] += Module_Center.Beta_mid[i*(M+1)+mm];
                tempt = Module_Center.Beta_mid[i*(M+1)+mm];
                sum_Beta2_mid[mm] += tempt * tempt;
                num_D[mm] ++;
            }
            sum_Data += Matrix1->data[i*numG + j];
        }
        tempt = sum_Beta_mid[mm] - sum_Data;
        SS.SS_md[mm] -= tempt* tempt/(numD + hyperpars.k_taod);

        sum_Data = 0.0;
        sum_Beta_mip[mm] = 0.0;
        sum_Beta2_mip[mm] = 0.0;
        num_P[mm] = 0;
        for (i = 0; i<numP; i++)
        {
            if(Indicator.Imp[i*(M+1)+mm] == 0)
            {
                tempt = Matrix2->data[i*numG + j];
                SS.SS_mp[mm] += tempt * tempt;
            }
            else
            {
                tempt = Matrix2->data[i*numG + j]- Module_Center.Beta_mip[i*(M+1)+mm];
                SS.SS_mp[mm] += tempt * tempt;
                sum_Beta_mip[mm] += Module_Center.Beta_mip[i*(M+1)+mm];
                tempt = Module_Center.Beta_mip[i*(M+1)+mm];
                sum_Beta2_mip[mm] += tempt * tempt;
                num_P[mm]++;
            }
            sum_Data += Matrix2->data[i*numG + j];
        }
        tempt = sum_Beta_mip[mm] - sum_Data;
        SS.SS_mp[mm] -= tempt*tempt/(numP + hyperpars.k_taop);
    }

    for (mm = 0; mm<=M;mm++)
    {
        sum_Beta_mid[mm] = 0.0;
        sum_Beta2_mid[mm] = 0.0;
        num_D[mm] = 0;
        for(i = 0; i<numD; i++)
        {
            if(Indicator.Imd[i*(M+1)+mm] == 1)
            {
                sum_Beta_mid[mm] += Module_Center.Beta_mid[i*(M+1)+mm];
                tempt = Module_Center.Beta_mid[i*(M+1)+mm];
                sum_Beta2_mid[mm] += tempt * tempt;
                num_D[mm] ++;
            }
        }

        sum_Beta_mip[mm] = 0.0;
        sum_Beta2_mip[mm] = 0.0;
        num_P[mm] = 0;
        for(i = 0; i<numP; i++)
        {
            if(Indicator.Imp[i*(M+1)+mm] == 1)
            {
                sum_Beta_mip[mm] += Module_Center.Beta_mip[i*(M+1)+mm];
                tempt = Module_Center.Beta_mip[i*(M+1)+mm];
                sum_Beta2_mip[mm] += tempt * tempt;
                num_P[mm] ++;
            }
        }

        SS.SS_mbd[mm] = sum_Beta2_mid[mm] - sum_Beta_mid[mm]*sum_Beta_mid[mm]/(num_D[mm] + hyperpars.k_sigmad);
        SS.SS_mbp[mm] = sum_Beta2_mip[mm] - sum_Beta_mip[mm]*sum_Beta_mip[mm]/(num_P[mm] + hyperpars.k_sigmap);
    }

}

void compute_Posterior(gsl_matrix* Matrix1,gsl_matrix* Matrix2,const struct_Indicator& Indicator, const struct_Module_Center& Module_Center)
{
    int mm,i,j;
    int M = Module_Count.M;
    double Posterior_part1 = 0.0;
    double SS_0d = 0.0;
    double SS_0p = 0.0;
    double tempt;

    double Matrix1_sum = 0.0;
    double Matrix2_sum = 0.0;
    double Matrix1_sum2 = 0.0;
    double Matrix2_sum2 = 0.0;
    int n_0G = 0;
    int n_mG[M+1];
    for (i = 0; i<=M; i++)
    {
        n_mG[i] = 0;
    }

    for(j = 0; j<numG; j++)
    {
        if(Indicator.Ig[j] !=0)
        {
            n_mG[Indicator.Ig[j]]++;
            continue;
        }

        n_0G ++;
        Matrix1_sum = 0.0;
        Matrix1_sum2 = 0.0;
        for(i = 0;i<numD;i++)
        {
            tempt = Matrix1->data[i*numG + j];
            Matrix1_sum += tempt;
            Matrix1_sum2 += tempt * tempt;
        }
        SS_0d += Matrix1_sum2 - Matrix1_sum*Matrix1_sum/(hyperpars.k_0d + numD);

        Matrix2_sum = 0.0;
        Matrix2_sum2 = 0.0;
        for(i = 0;i<numP;i++)
        {
            tempt = Matrix2->data[i*numG + j];
            Matrix2_sum += tempt;
            Matrix2_sum2 += tempt * tempt;
        }
        SS_0p += Matrix2_sum2 - Matrix2_sum*Matrix2_sum/(hyperpars.k_0p + numP);
    }

    double v_td = hyperpars.v_0d/2;
    double v_newtd = (hyperpars.v_0d + n_0G*numD)/2;
    Posterior_part1 += gsl_sf_log(hyperpars.k_0d/(hyperpars.k_0d + numD))*n_0G/2\
                       + gsl_sf_log(hyperpars.v_0d*hyperpars.S2_0d)*v_td\
                       + gsl_sf_lngamma(v_newtd) - gsl_sf_lngamma(v_td)\
                       - gsl_sf_log(hyperpars.v_0d*hyperpars.S2_0d + SS_0d)*v_newtd;
    double v_tp = hyperpars.v_0p/2;
    double v_newtp = (hyperpars.v_0p + n_0G*numP)/2;
    Posterior_part1 += gsl_sf_log(hyperpars.k_0p/(hyperpars.k_0p + numP))*n_0G/2\
                       + gsl_sf_log(hyperpars.v_0p*hyperpars.S2_0p)*v_tp\
                       + gsl_sf_lngamma(v_newtp) - gsl_sf_lngamma(v_tp)\
                       - gsl_sf_log(hyperpars.v_0p*hyperpars.S2_0p + SS_0p)*v_newtp;


    double Posterior_part2 = 0.0;
    double Posterior_part3 = 0.0;
    for (mm = 1; mm<=M; mm++)
    {
        v_td = hyperpars.v_taod/2;
        v_newtd = (hyperpars.v_taod + numD*n_mG[mm])/2;
        Posterior_part2 += gsl_sf_log(hyperpars.k_taod/(numD + hyperpars.k_taod))*n_mG[mm]/2\
                           + gsl_sf_log(hyperpars.v_taod*hyperpars.S2_taod)*v_td\
                           + gsl_sf_lngamma(v_newtd) - gsl_sf_lngamma(v_td)\
                           - gsl_sf_log(hyperpars.v_taod*hyperpars.S2_taod + SS.SS_md[mm])*v_newtd;

        v_tp = hyperpars.v_taop/2;
        v_newtp = (hyperpars.v_taop + numP*n_mG[mm])/2;
        Posterior_part2 += gsl_sf_log(hyperpars.k_taop/(numP + hyperpars.k_taop))*n_mG[mm]/2\
                          + gsl_sf_log(hyperpars.v_taop*hyperpars.S2_taop)*v_tp\
                          + gsl_sf_lngamma(v_newtp) - gsl_sf_lngamma(v_tp)\
                          - gsl_sf_log(hyperpars.v_taop*hyperpars.S2_taop + SS.SS_mp[mm])*v_newtp;

        int n_mD = 0;
        int n_mP = 0;
        for(i = 0;i<numD;i++)
        {
            if(Indicator.Imd[i*(M+1)+mm] ==1)
                n_mD ++;
        }
        v_td = hyperpars.v_sigmad/2;
        v_newtd = (n_mD + hyperpars.v_sigmad)/2;
        Posterior_part3 += M_LNPI*(-n_mD*1.0/2) + gsl_sf_log(hyperpars.k_sigmad/(n_mD + hyperpars.k_sigmad))*0.5\
                           + gsl_sf_log(hyperpars.v_sigmad*hyperpars.S2_sigmad)*v_td\
                           + gsl_sf_lngamma(v_newtd) - gsl_sf_lngamma(v_td)\
                           - gsl_sf_log(hyperpars.v_sigmad*hyperpars.S2_sigmad + SS.SS_mbd[mm])*v_newtd;

        for(i = 0;i<numP;i++)
        {
            if(Indicator.Imp[i*(M+1)+mm] ==1)
                n_mP ++;
        }
        v_tp = hyperpars.v_sigmap/2;
        v_newtp = (n_mP + hyperpars.v_sigmap)/2;
        Posterior_part3 += M_LNPI*(-n_mP*1.0/2) + gsl_sf_log(hyperpars.k_sigmap/(n_mP + hyperpars.k_sigmap))*0.5\
                           + gsl_sf_log(hyperpars.v_sigmap*hyperpars.S2_sigmap)*v_tp\
                           + gsl_sf_lngamma(v_newtp) - gsl_sf_lngamma(v_tp)\
                           - gsl_sf_log(hyperpars.v_sigmap*hyperpars.S2_sigmap + SS.SS_mbp[mm])*v_newtp;
    }

    Posterior.part1 = Posterior_part1;
    Posterior.part2 = Posterior_part2;
    Posterior.part3 = Posterior_part3;

}

void compute_Sum_Num(struct_Indicator& Indicator)
{
    int M = Module_Count.M;
    int mm,i;
    Sum_Num.mG = 0;
    Sum_Num.mD = 0;
    Sum_Num.mP = 0;
    for (i = 0; i<numG; i++)
    {
        if(Indicator.Ig[i] != 0)
        {
            Sum_Num.mG++;
        }
    }
    for(mm = 1;mm<=M; mm++)
    {
        for(i = 0; i<numD;i++)
        {
            if(Indicator.Imd[i*(M+1) + mm] == 1)
                Sum_Num.mD ++;
        }
        for(i = 0; i<numP; i++)
        {
            if(Indicator.Imp[i*(M+1) + mm] == 1)
                Sum_Num.mP ++;
        }
    }
}

void compute_Sum_Num_local(int* Ig, int* Imd, int* Imp,struct_Sum_Num& Sum_Num_local)
{
    int M = Module_Count.M;
    int i,mm;
    Sum_Num_local.mG = 0;
    Sum_Num_local.mD = 0;
    Sum_Num_local.mP = 0;
    for (i = 0; i<numG; i++)
    {
        if(Ig[i] != 0)
        {
            Sum_Num_local.mG++;
        }
    }
    for(mm = 1;mm<=M; mm++)
    {
        for(i = 0; i<numD;i++)
        {
            if(Imd[i*(M+1) + mm] == 1)
                Sum_Num_local.mD ++;
        }
        for(i = 0; i<numP; i++)
        {
            if(Imp[i*(M+1) + mm] == 1)
                Sum_Num_local.mP ++;
        }
    }
}

void compute_num_G(const int* Ig, int* num_G, int M)
{
    register int i;
    for(i = 0; i<=M; i++)
        num_G[i] = 0;
    for(i = 0; i<numG; i++)
        num_G[Ig[i]]++;
}

//========================================================
void change_SS_md(gsl_matrix* Matrix1,int* Ig,int* Imd,double* Beta_mid,double* SS_md_new,int mm)
{
    int i,j;
    int M = Module_Count.M;
    double tempt;
    SS_md_new[mm] = 0.0;

    double sum_Beta_mid = 0.0;
    double sum_Data = 0.0;
    for(j = 0; j<numG;j++)
    {
        if(Ig[j] != mm)
            continue;
        sum_Beta_mid = 0.0;
        sum_Data = 0.0;
        for(i = 0; i<numD;i++)
        {
            if(Imd[i*(M+1)+mm] == 0)
            {
                tempt = Matrix1->data[i*numG + j];
                SS_md_new[mm] += tempt * tempt;
            }
            else
            {
                tempt = Matrix1->data[i*numG + j] - Beta_mid[i*(M+1)+mm];
                SS_md_new[mm] += tempt * tempt;
                sum_Beta_mid += Beta_mid[i*(M+1)+mm];
            }
            sum_Data += Matrix1->data[i*numG + j];
        }
        SS_md_new[mm] -= (sum_Beta_mid - sum_Data)*(sum_Beta_mid - sum_Data)/(numD + hyperpars.k_taod);
    }

}

void change_SS_mp(gsl_matrix* Matrix2,int* Ig,int* Imp,double* Beta_mip,double* SS_mp_new,int mm)
{
    int i,j;
    int M = Module_Count.M;
    SS_mp_new[mm] = 0.0;
    double tempt;

    double sum_Beta_mip = 0.0;
    double sum_Data = 0.0;
    for(j = 0; j<numG;j++)
    {
        if(Ig[j] != mm)
            continue;
        sum_Beta_mip = 0.0;
        sum_Data = 0.0;
        for(i = 0; i<numP;i++)
        {
            if(Imp[i*(M+1)+mm] == 0)
            {
                tempt = Matrix2->data[i*numG + j];
                SS_mp_new[mm] += tempt * tempt;
            }
            else
            {
                tempt = Matrix2->data[i*numG + j] - Beta_mip[i*(M+1)+mm];
                SS_mp_new[mm] += tempt * tempt;
                sum_Beta_mip += Beta_mip[i*(M+1)+mm];
            }
            sum_Data += Matrix2->data[i*numG + j];
        }
        SS_mp_new[mm] -= (sum_Beta_mip - sum_Data)*(sum_Beta_mip - sum_Data)/(numP + hyperpars.k_taop);
    }
}

void change_SS_mbd(int* Imd,double* Beta_mid,double* SS_mbd_new,int mm)
{
    int M = Module_Count.M;
    SS_mbd_new[mm] = 0.0;
    int i;

    double sum_Beta_mid = 0.0;
    double sum_Beta2_mid = 0.0;
    int num_D = 0;
    for(i= 0; i<numD;i++)
    {
        if(Imd[i*(M+1)+mm] == 1)
        {
            sum_Beta_mid += Beta_mid[i*(M+1)+mm];
            sum_Beta2_mid += Beta_mid[i*(M+1)+mm]*Beta_mid[i*(M+1)+mm];
            num_D++;
        }
    }
    SS_mbd_new[mm] = sum_Beta2_mid - sum_Beta_mid*sum_Beta_mid/(num_D + hyperpars.k_sigmad);
}

void change_SS_mbp(int* Imp,double* Beta_mip,double* SS_mbp_new,int mm)
{
    int M = Module_Count.M;
    SS_mbp_new[mm] = 0.0;
    int i;
    double sum_Beta_mip = 0.0;
    double sum_Beta2_mip = 0.0;
    int num_P = 0;
    for(i= 0; i<numP;i++)
    {
        if(Imp[i*(M+1)+mm] == 1)
        {
            sum_Beta_mip += Beta_mip[i*(M+1)+mm];
            sum_Beta2_mip += Beta_mip[i*(M+1)+mm]*Beta_mip[i*(M+1)+mm];
            num_P++;
        }
    }
    SS_mbp_new[mm] = sum_Beta2_mip - sum_Beta_mip*sum_Beta_mip/(num_P + hyperpars.k_sigmap);
}

double change_Posterior_part3(double Posterior_part3,int* Imd, int* Imd_new, int* Imp, int* Imp_new,\
                              double* SS_mbd, double* SS_mbd_new, double* SS_mbp, double* SS_mbp_new,int mm)
{
    int M = Module_Count.M;
    int n_mD, n_mD_new, n_mP,n_mP_new;
    int i;
    n_mD = 0;
    n_mD_new = 0;
    n_mP = 0;
    n_mP_new = 0;
    for(i = 0; i<numD; i++)
    {
        if(Imd[i*(M+1)+mm] == 1)
            n_mD ++;
        if(Imd_new[i*(M+1)+mm] == 1)
            n_mD_new++;
    }
    for(i = 0; i<numP; i++)
    {
        if(Imp[i*(M+1)+mm] == 1)
            n_mP ++;
        if(Imp_new[i*(M+1)+mm] == 1)
            n_mP_new++;
    }

    double Posterior_part3_new = Posterior_part3;

    double v_newtd = (n_mD + hyperpars.v_sigmad)/2;
    Posterior_part3_new -= M_LNPI*(-n_mD*1.0/2) + gsl_sf_log(hyperpars.k_sigmad/(n_mD + hyperpars.k_sigmad))*0.5\
                           + gsl_sf_lngamma(v_newtd) - gsl_sf_log(hyperpars.v_sigmad*hyperpars.S2_sigmad + SS_mbd[mm])*v_newtd;
    v_newtd = (n_mD_new + hyperpars.v_sigmad)/2;
    Posterior_part3_new += M_LNPI*(-n_mD_new*1.0/2) + gsl_sf_log(hyperpars.k_sigmad/(n_mD_new + hyperpars.k_sigmad))*0.5\
                           + gsl_sf_lngamma(v_newtd) - gsl_sf_log(hyperpars.v_sigmad*hyperpars.S2_sigmad + SS_mbd_new[mm])*v_newtd;

    double v_newtp = (n_mP + hyperpars.v_sigmap)/2;
    Posterior_part3_new -= M_LNPI*(-n_mP*1.0/2) + gsl_sf_log(hyperpars.k_sigmap/(n_mP + hyperpars.k_sigmap))*0.5\
                           + gsl_sf_lngamma(v_newtp) - gsl_sf_log(hyperpars.v_sigmap*hyperpars.S2_sigmap + SS_mbp[mm])*v_newtp;
    v_newtp = (n_mP_new + hyperpars.v_sigmap)/2;
    Posterior_part3_new += M_LNPI*(-n_mP_new*1.0/2) + gsl_sf_log(hyperpars.k_sigmap/(n_mP_new + hyperpars.k_sigmap))*0.5\
                           + gsl_sf_lngamma(v_newtp) - gsl_sf_log(hyperpars.v_sigmap*hyperpars.S2_sigmap + SS_mbp_new[mm])*v_newtp;

    return Posterior_part3_new;


}

double change_Posterior_part2(gsl_matrix* Matrix1, gsl_matrix* Matrix2,double Posterior_part2,int* Ig, int* Ig_new,\
                              double* SS_md,double* SS_md_new,double* SS_mp,double* SS_mp_new,int mm)
{
    int i;
    int n_mG_m = 0;
    int n_mG_m_new = 0;
    for(i = 0; i<numG; i++)
    {
        if(Ig[i] == mm)
            n_mG_m++;
        if(Ig_new[i] == mm)
            n_mG_m_new++;
    }

    double Posterior_part2_new = Posterior_part2;
    Posterior_part2_new -= ( gsl_sf_log(hyperpars.k_taod/(numD + hyperpars.k_taod))*(n_mG_m*1.0/2) + gsl_sf_lngamma((numD*n_mG_m + hyperpars.v_taod)/2)\
                              - gsl_sf_log(hyperpars.v_taod*hyperpars.S2_taod + SS_md[mm])*(numD*n_mG_m + hyperpars.v_taod)/2  );
    Posterior_part2_new += ( gsl_sf_log(hyperpars.k_taod/(numD + hyperpars.k_taod))*(n_mG_m_new*1.0/2) + gsl_sf_lngamma((numD*n_mG_m_new + hyperpars.v_taod)/2)\
                             - gsl_sf_log(hyperpars.v_taod*hyperpars.S2_taod + SS_md_new[mm])*(numD*n_mG_m_new + hyperpars.v_taod)/2  );
    Posterior_part2_new -= ( gsl_sf_log(hyperpars.k_taop/(numP + hyperpars.k_taop))*(n_mG_m*1.0/2) + gsl_sf_lngamma((numP*n_mG_m + hyperpars.v_taop)/2)\
                              - gsl_sf_log(hyperpars.v_taop*hyperpars.S2_taop + SS_mp[mm])*(numP*n_mG_m + hyperpars.v_taop)/2  );
    Posterior_part2_new += ( gsl_sf_log(hyperpars.k_taop/(numP + hyperpars.k_taop))*(n_mG_m_new*1.0/2) + gsl_sf_lngamma((numP*n_mG_m_new + hyperpars.v_taop)/2)\
                              - gsl_sf_log(hyperpars.v_taop*hyperpars.S2_taop + SS_mp_new[mm])*(numP*n_mG_m_new + hyperpars.v_taop)/2  );
    return Posterior_part2_new;
}

double change_Posterior_part1(gsl_matrix* Matrix1, gsl_matrix* Matrix2, int* Ig)
{
    int M = Module_Count.M;
    double Posterior_part1 = 0.0;
    double SS_0d = 0.0;
    double SS_0p = 0.0;
    int i,j;

    double Matrix1_sum = 0.0;
    double Matrix2_sum = 0.0;
    double Matrix1_sum2 = 0.0;
    double Matrix2_sum2 = 0.0;
    int n_0G = 0;

    for(j = 0; j<numG; j++)
    {
        if(Ig[j] !=0)
        {
            continue;
        }

        n_0G ++;
        Matrix1_sum = 0.0;
        Matrix1_sum2 = 0.0;
        for(i = 0;i<numD;i++)
        {
            Matrix1_sum += Matrix1->data[i*numG + j];
            Matrix1_sum2 += Matrix1->data[i*numG + j]*Matrix1->data[i*numG + j];
        }
        SS_0d += Matrix1_sum2 - Matrix1_sum*Matrix1_sum/(hyperpars.k_0d + numD);

        Matrix2_sum = 0.0;
        Matrix2_sum2 = 0.0;
        for(i = 0;i<numP;i++)
        {
            Matrix2_sum += Matrix2->data[i*numG + j];
            Matrix2_sum2 += Matrix2->data[i*numG + j]*Matrix2->data[i*numG + j];
        }
        SS_0p += Matrix2_sum2 - Matrix2_sum*Matrix2_sum/(hyperpars.k_0p + numP);
    }

    double v_td = hyperpars.v_0d/2;
    double v_newtd = (hyperpars.v_0d + n_0G*numD)/2;
    Posterior_part1 += gsl_sf_log(hyperpars.k_0d/(hyperpars.k_0d + numD))*n_0G/2\
                       + gsl_sf_log(hyperpars.v_0d*hyperpars.S2_0d)*v_td\
                       + gsl_sf_lngamma(v_newtd) - gsl_sf_lngamma(v_td)\
                       - gsl_sf_log(hyperpars.v_0d*hyperpars.S2_0d + SS_0d)*v_newtd;
    double v_tp = hyperpars.v_0p/2;
    double v_newtp = (hyperpars.v_0p + n_0G*numP)/2;
    Posterior_part1 += gsl_sf_log(hyperpars.k_0p/(hyperpars.k_0p + numP))*n_0G/2\
                       + gsl_sf_log(hyperpars.v_0p*hyperpars.S2_0p)*v_tp\
                       + gsl_sf_lngamma(v_newtp) - gsl_sf_lngamma(v_tp)\
                       - gsl_sf_log(hyperpars.v_0p*hyperpars.S2_0p + SS_0p)*v_newtp;

    return Posterior_part1;
}

double Compute_Total_Posterior(struct_Sum_Num& Sum_Num_local,struct_Posterior& Posterior_local,int M)
{
    double Total_Posterior;
    Total_Posterior = Posterior_local.part1 + Posterior_local.part2 + Posterior_local.part3\
                      - hyperpars.C_D * Sum_Num_local.mD\
                      - hyperpars.C_P * Sum_Num_local.mP\
                      - hyperpars.C_G * Sum_Num_local.mG\
                      - hyperpars.C_M * M;
    return Total_Posterior*Scale;
}
//======================================================
void Gene_Indicator_Propose_to_Module(gsl_matrix* Matrix1, gsl_matrix* Matrix2, struct_Indicator& Indicator, struct_Module_Center& Module_Center, int ii, int mm,\
                                      struct_SS* SS_new, struct_Sum_Num* Sum_Num_new, struct_Posterior* Posterior_new)
{
    int M = Module_Count.M;
    int mmi,i;

    int Ig_new[numG];
    int curr_m = Indicator.Ig[ii];
    for(i = 0;i<numG;i++)
    {
        Ig_new[i] = Indicator.Ig[i];
    }
    Ig_new[ii] = mm;

    /*for(mmi = 0; mmi<= M; mmi++)
    {
        SS_new->SS_md[mmi] = SS.SS_md[mmi];
        SS_new->SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new->SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new->SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    Posterior_new->part1 = Posterior.part1;
    Posterior_new->part2 = Posterior.part2;
    Posterior_new->part3 = Posterior.part3;
    Sum_Num_new->mD = Sum_Num.mD;
    Sum_Num_new->mG = Sum_Num.mG;
    Sum_Num_new->mP = Sum_Num.mP;

    if(curr_m != 0)
    {
        change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new->SS_md,mm);
        change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new->SS_md,curr_m);
        change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new->SS_mp,mm);
        change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new->SS_mp,curr_m);

        Posterior_new->part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new->part2,Indicator.Ig,Ig_new\
                                                      ,SS.SS_md,SS_new->SS_md,SS.SS_mp,SS_new->SS_mp,mm);
        Posterior_new->part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new->part2,Indicator.Ig,Ig_new\
                                                      ,SS.SS_md,SS_new->SS_md,SS.SS_mp,SS_new->SS_mp,curr_m);
    }
    else
    {
        Sum_Num_new->mG ++;
        change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new->SS_md,mm);
        change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new->SS_mp,mm);

        Posterior_new->part1 = change_Posterior_part1(Matrix1,Matrix2,Ig_new);
        Posterior_new->part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new->part2,Indicator.Ig,Ig_new\
                                                      ,SS.SS_md,SS_new->SS_md,SS.SS_mp,SS_new->SS_mp,mm);
    }


}

void Gene_Indicator_Propose_to_NULL(gsl_matrix* Matrix1, gsl_matrix* Matrix2, struct_Indicator& Indicator, struct_Module_Center& Module_Center, int ii,\
                                    struct_SS* SS_new, struct_Sum_Num* Sum_Num_new, struct_Posterior* Posterior_new)
{
    int M = Module_Count.M;
    int i,mmi;

    int Ig_new[numG];
    int curr_m = Indicator.Ig[ii];
    for(i = 0;i<numG;i++)
    {
        Ig_new[i] = Indicator.Ig[i];
    }
    Ig_new[ii] = 0;

    /*for(mmi = 0; mmi<= M; mmi++)
    {
        SS_new->SS_md[mmi] = SS.SS_md[mmi];
        SS_new->SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new->SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new->SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    Posterior_new->part1 = Posterior.part1;
    Posterior_new->part2 = Posterior.part2;
    Posterior_new->part3 = Posterior.part3;

    Sum_Num_new->mD = Sum_Num.mD;
    Sum_Num_new->mG = Sum_Num.mG;
    Sum_Num_new->mP = Sum_Num.mP;

    if(curr_m != 0)
    {
        change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new->SS_md,curr_m);
        change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new->SS_mp,curr_m);

        Posterior_new->part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new->part2,Indicator.Ig,Ig_new\
                                                      ,SS.SS_md,SS_new->SS_md,SS.SS_mp,SS_new->SS_mp,curr_m);
        Posterior_new->part1 = change_Posterior_part1(Matrix1,Matrix2,Ig_new);
    }
    Sum_Num_new->mG--;
}

void Sample_Gene_Indicator(gsl_matrix* Matrix1,gsl_matrix* Matrix2,struct_Indicator& Indicator,struct_Module_Center& Module_Center,int ii,\
                           gsl_rng* r)
{
    int M = Module_Count.M;
    int mm;
    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    struct_SS SS_Array[M+1];
    struct_Sum_Num Sum_Num_Array[M+1];
    struct_Posterior Posterior_Array[M+1];
    for(mm = 0; mm<=M; mm++)
    {
        SS_Array[mm].SS_md = (double*)malloc((M+1)*sizeof(double));
        SS_Array[mm].SS_mp = (double*)malloc((M+1)*sizeof(double));
        SS_Array[mm].SS_mbd = (double*)malloc((M+1)*sizeof(double));
        SS_Array[mm].SS_mbp = (double*)malloc((M+1)*sizeof(double));

    }
    int curr_m = Indicator.Ig[ii];

    Sum_Num_Array[curr_m] = Sum_Num;
    Posterior_Array[curr_m] = Posterior;
    //------------------------------------------------------------
    double Gibbs_Vector[M+1];
    Gibbs_Vector[curr_m] = total_Posterior;
    if(curr_m!=0)
    {
        Gene_Indicator_Propose_to_NULL(Matrix1,Matrix2,Indicator,Module_Center,ii,&SS_Array[0],&Sum_Num_Array[0],&Posterior_Array[0]);
        Gibbs_Vector[0] = Compute_Total_Posterior(Sum_Num_Array[0],Posterior_Array[0],M);
    }

    for(mm=1;mm<=M;mm++)
    {
        if(mm == curr_m)
            continue;
        Gene_Indicator_Propose_to_Module(Matrix1,Matrix2,Indicator,Module_Center,ii,mm,&SS_Array[mm],&Sum_Num_Array[mm],&Posterior_Array[mm]);
        Gibbs_Vector[mm] = Compute_Total_Posterior(Sum_Num_Array[mm],Posterior_Array[mm],M);
    }



    double Max_Vector = 0.0;
    double Sum_Vector = 0.0;
    double Cumulate_P[M+1];
    for (mm = 0; mm<=M; mm++)
    {
        if(Max_Vector<Gibbs_Vector[mm])
            Max_Vector = Gibbs_Vector[mm];
    }
    for(mm = 0; mm<=M; mm++)
    {
        if(Gibbs_Vector[mm]-Max_Vector > -500.0)
        {
            Gibbs_Vector[mm] = gsl_sf_exp(Gibbs_Vector[mm] - Max_Vector);
        }
        else
            Gibbs_Vector[mm] = 0.0;
        Sum_Vector += Gibbs_Vector[mm];
    }
    Cumulate_P[0] = Gibbs_Vector[0]/Sum_Vector;
    for(mm = 1; mm<=M; mm++)
    {
        Cumulate_P[mm] = Cumulate_P[mm-1] + Gibbs_Vector[mm]/Sum_Vector;
    }

    double rand_t = gsl_rng_uniform (r);
    int i = 0;
    while(i <= M)
    {
        if(rand_t <= Cumulate_P[i])
        {
            Indicator.Ig[ii] = i;
            break;
        }
        else
            i++;
    }

    for(mm = 0; mm<=M; mm++)
    {
        if(mm == i && i != curr_m)
        {
            SS.SS_md[curr_m] = SS_Array[mm].SS_md[curr_m];
            SS.SS_mp[curr_m] = SS_Array[mm].SS_mp[curr_m];
            SS.SS_md[mm] = SS_Array[mm].SS_md[mm];
            SS.SS_mp[mm] = SS_Array[mm].SS_mp[mm];

            Sum_Num = Sum_Num_Array[mm];
            Posterior = Posterior_Array[mm];
        }
        free(SS_Array[mm].SS_md) ;
        free(SS_Array[mm].SS_mbd);
        free(SS_Array[mm].SS_mbp);
        free(SS_Array[mm].SS_mp);
    }
}

void Drug_Indicator_Propose(gsl_matrix* Matrix1,gsl_matrix* Matrix2,struct_Indicator& Indicator, struct_Module_Center& Module_Center,int ii,int mm,\
                            struct_SS* SS_new, struct_Sum_Num* Sum_Num_new, struct_Posterior* Posterior_new)
{

    int M = Module_Count.M;
    //int Imd_new[numD*(M+1)];
    int* Imd_new = (int*)malloc(sizeof(int)*numD*(M+1));
    int i,mmi;
    for(i = 0; i<numD; i++)
    {
        for(mmi = 0; mmi<=M; mmi++)
            Imd_new[i*(M+1)+mmi] = Indicator.Imd[i*(M+1)+mmi];
    }
    Imd_new[ii*(M+1) + mm] = 1-Indicator.Imd[ii*(M+1)+mm];

    /*for(mmi = 0; mmi<= M; mmi++)
    {
        SS_new->SS_md[mmi] = SS.SS_md[mmi];
        SS_new->SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new->SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new->SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    Posterior_new->part1 = Posterior.part1;
    Posterior_new->part2 = Posterior.part2;
    Posterior_new->part3 = Posterior.part3;

    Sum_Num_new->mD = Sum_Num.mD;
    Sum_Num_new->mG = Sum_Num.mG;
    Sum_Num_new->mP = Sum_Num.mP;

    //============================================
    change_SS_md(Matrix1,Indicator.Ig,Imd_new,Module_Center.Beta_mid,SS_new->SS_md,mm);
    change_SS_mbd(Imd_new,Module_Center.Beta_mid,SS_new->SS_mbd,mm);

    Posterior_new->part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new->part2,Indicator.Ig,Indicator.Ig\
                                                  ,SS.SS_md,SS_new->SS_md,SS.SS_mp,SS.SS_mp,mm);
    Posterior_new->part3 = change_Posterior_part3(Posterior_new->part3,Indicator.Imd,Imd_new,Indicator.Imp,Indicator.Imp,\
                                                  SS.SS_mbd,SS_new->SS_mbd,SS.SS_mbp,SS.SS_mbp,mm);
    //============================================
    if(Imd_new[ii*(M+1) + mm] == 1)
        Sum_Num_new->mD ++;
    else
        Sum_Num_new->mD --;

    free(Imd_new);



}

void Sample_Drug_Indicator(gsl_matrix* Matrix1,gsl_matrix* Matrix2,\
                           struct_Indicator& Indicator,struct_Module_Center& Module_Center,int input_mm,\
                           gsl_rng* r)
{
    int i;
    int M = Module_Count.M;
    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double Gibbs_Vector[2];

    struct_SS SS_new;
    SS_new.SS_md = (double*)malloc(sizeof(double)*(M+1));
    //SS_new.SS_mp = (double*)malloc(sizeof(double)*(M+1));
    SS_new.SS_mbd = (double*)malloc(sizeof(double)*(M+1));
    //SS_new.SS_mbp = (double*)malloc(sizeof(double)*(M+1));

    struct_Sum_Num Sum_Num_new;
    struct_Posterior Posterior_new;

    for(i=0; i<numD; i++)
    {
        int curr_di = Indicator.Imd[i*(M+1) + input_mm];
        Gibbs_Vector[curr_di] = total_Posterior;
        //==================================
        Drug_Indicator_Propose(Matrix1,Matrix2,Indicator,Module_Center,i,input_mm,\
                               &SS_new, &Sum_Num_new,&Posterior_new);
        Gibbs_Vector[1-curr_di] = Compute_Total_Posterior(Sum_Num_new,Posterior_new,M);
        //==================================

        double Total_Posterior_Array[2];
        Total_Posterior_Array[0] = Gibbs_Vector[0];
        Total_Posterior_Array[1] = Gibbs_Vector[1];

        double Max_Vector = 0.0;
        if(Gibbs_Vector[0]>Gibbs_Vector[1])
            Max_Vector = Gibbs_Vector[0];
        else
            Max_Vector = Gibbs_Vector[1];

        double Sum_Vector = 0.0;
        double Cumulate_P[2];
        for(int ii = 0; ii<2; ii++)
        {
            if(Gibbs_Vector[ii]-Max_Vector > -500.0)
            {
                Gibbs_Vector[ii] = gsl_sf_exp(Gibbs_Vector[ii] - Max_Vector);
            }
            else
                Gibbs_Vector[ii] = 0.0;
            Sum_Vector += Gibbs_Vector[ii];
        }
        Cumulate_P[0] = Gibbs_Vector[0]/Sum_Vector;
        Cumulate_P[1] = 1;
        double rand_t = gsl_rng_uniform (r);
        //rand_t = rand_number_collection.front();
        //rand_number_collection.pop();

        if(rand_t <= Cumulate_P[0])
            Indicator.Imd[i*(M+1)+input_mm] = 0;
        else
            Indicator.Imd[i*(M+1)+input_mm] = 1;

        if(Indicator.Imd[i*(M+1)+input_mm] != curr_di)
        {
            SS.SS_md[input_mm] = SS_new.SS_md[input_mm];
            SS.SS_mbd[input_mm] = SS_new.SS_mbd[input_mm];

            Sum_Num = Sum_Num_new;
            Posterior = Posterior_new;

            total_Posterior = Total_Posterior_Array[1-curr_di];
        }
    }
    free(SS_new.SS_mbd);
    //free(SS_new.SS_mbp);
    free(SS_new.SS_md);
    //free(SS_new.SS_mp);
    //gsl_rng_free(r);
}

void Phenotype_Indicator_Propose(gsl_matrix* Matrix1,gsl_matrix* Matrix2,struct_Indicator& Indicator, struct_Module_Center& Module_Center,int ii,int mm,\
                            struct_SS* SS_new, struct_Sum_Num* Sum_Num_new, struct_Posterior* Posterior_new)
{
    int M = Module_Count.M;
    //int Imp_new[numP*(M+1)];
    int* Imp_new = (int*)malloc(sizeof(int)*numP*(M+1));
    int i,mmi;
    for(i = 0; i<numP; i++)
    {
        for(mmi = 0; mmi<=M; mmi++)
            Imp_new[i*(M+1)+mmi] = Indicator.Imp[i*(M+1)+mmi];
    }
    Imp_new[ii*(M+1) + mm] = 1-Indicator.Imp[ii*(M+1)+mm];

    Posterior_new->part1 = Posterior.part1;
    Posterior_new->part2 = Posterior.part2;
    Posterior_new->part3 = Posterior.part3;

    Sum_Num_new->mD = Sum_Num.mD;
    Sum_Num_new->mG = Sum_Num.mG;
    Sum_Num_new->mP = Sum_Num.mP;

    //============================================
    change_SS_mp(Matrix2,Indicator.Ig,Imp_new,Module_Center.Beta_mip,SS_new->SS_mp,mm);
    change_SS_mbp(Imp_new,Module_Center.Beta_mip,SS_new->SS_mbp,mm);

    Posterior_new->part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new->part2,Indicator.Ig,Indicator.Ig\
                                                  ,SS.SS_md,SS.SS_md,SS.SS_mp,SS_new->SS_mp,mm);
    Posterior_new->part3 = change_Posterior_part3(Posterior_new->part3,Indicator.Imd,Indicator.Imd,Indicator.Imp,Imp_new,\
                                                  SS.SS_mbd,SS.SS_mbd,SS.SS_mbp,SS_new->SS_mbp,mm);
    //============================================
    if(Imp_new[ii*(M+1) + mm] == 1)
        Sum_Num_new->mP ++;
    else
        Sum_Num_new->mP --;

    free(Imp_new);
}

void Sample_Phenotype_Indicator(gsl_matrix* Matrix1,gsl_matrix* Matrix2,\
                                struct_Indicator& Indicator,struct_Module_Center& Module_Center,int input_mm,\
                                gsl_rng* r)
{
    int i;
    int M = Module_Count.M;

    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double Gibbs_Vector[2];

    struct_SS SS_new; // to record the new proposed global variables
    struct_Sum_Num Sum_Num_new;
    struct_Posterior Posterior_new;
    //SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));

    for(i=0; i<numP; i++)
    {
        int curr_pi = Indicator.Imp[i*(M+1) + input_mm];
        Gibbs_Vector[curr_pi] = total_Posterior;
        //==================================
        Phenotype_Indicator_Propose(Matrix1,Matrix2,Indicator,Module_Center,i,input_mm,\
                               &SS_new, &Sum_Num_new,&Posterior_new);
        Gibbs_Vector[1-curr_pi] = Compute_Total_Posterior(Sum_Num_new,Posterior_new,M);

        //==================================

        double Total_Posterior_Array[2];
        Total_Posterior_Array[0] = Gibbs_Vector[0];
        Total_Posterior_Array[1] = Gibbs_Vector[1];

        double Max_Vector = 0.0;
        if(Gibbs_Vector[0]>Gibbs_Vector[1])
            Max_Vector = Gibbs_Vector[0];
        else
            Max_Vector = Gibbs_Vector[1];

        double Sum_Vector = 0.0;
        double Cumulate_P[2];
        for(int ii = 0; ii<2; ii++)
        {
            if(Gibbs_Vector[ii]-Max_Vector > -500.0)
            {
                Gibbs_Vector[ii] = gsl_sf_exp(Gibbs_Vector[ii] - Max_Vector);
            }
            else
                Gibbs_Vector[ii] = 0.0;
            Sum_Vector += Gibbs_Vector[ii];
        }
        Cumulate_P[0] = Gibbs_Vector[0]/Sum_Vector;
        Cumulate_P[1] = 1;
        double rand_t = gsl_rng_uniform (r);
        //rand_t = rand_number_collection.front();
        //rand_number_collection.pop();
        if(rand_t <= Cumulate_P[0])
            Indicator.Imp[i*(M+1)+input_mm] = 0;
        else
            Indicator.Imp[i*(M+1)+input_mm] = 1;

        if(Indicator.Imp[i*(M+1)+input_mm] != curr_pi)
        {
            SS.SS_mp[input_mm] = SS_new.SS_mp[input_mm];
            SS.SS_mbp[input_mm] = SS_new.SS_mbp[input_mm];

            Sum_Num = Sum_Num_new;
            Posterior = Posterior_new;

            total_Posterior = Total_Posterior_Array[1-curr_pi];
        }
    }
    //free(SS_new.SS_mbd);
    free(SS_new.SS_mbp);
    //free(SS_new.SS_md);
    free(SS_new.SS_mp);
    //gsl_rng_free(r);
}

//====================================================
void Change_Drug_Module_Center(gsl_matrix* Matrix1,gsl_matrix* Matrix2,\
                               struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                               int* num_G, gsl_rng* r)
{
    int i;
    int M = Module_Count.M;

    struct_SS SS_new;
    struct_Sum_Num Sum_Num_new;
    struct_Posterior Posterior_new;
    SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    int mmi,mm;

    /*for(mmi = 0;mmi<=M;mmi++)
    {
        SS_new.SS_md[mmi] = SS.SS_md[mmi];
        SS_new.SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new.SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new.SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;
    double pre_module_center;

    for(mm=1; mm<=M; mm++)
    {
        if(num_G[mm] ==0)
            continue;
        for(i = 0; i<numD; i++)
        {
            if(Indicator.Imd[i*(M+1)+mm] != 1)
                continue;
            Sum_Num_new = Sum_Num;
            Posterior_new = Posterior;

            rand_t = gsl_rng_uniform (r);
            //rand_t = rand_number_collection.front();
            //rand_number_collection.pop();
            pre_module_center = Module_Center.Beta_mid[i*(M+1) + mm];
            if(rand_t<= 0.5)
                Module_Center.Beta_mid[i*(M+1) + mm] += Delta_Beta;
            else
                Module_Center.Beta_mid[i*(M+1) + mm] -= Delta_Beta;

            //======================================================
            change_SS_md(Matrix1,Indicator.Ig,Indicator.Imd,Module_Center.Beta_mid,SS_new.SS_md,mm);
            change_SS_mbd(Indicator.Imd,Module_Center.Beta_mid,SS_new.SS_mbd,mm);

            Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Indicator.Ig,\
                                                         SS.SS_md,SS_new.SS_md,SS.SS_mp,SS.SS_mp,mm);
            Posterior_new.part3 = change_Posterior_part3(Posterior_new.part3,Indicator.Imd,Indicator.Imd,Indicator.Imp,Indicator.Imp,\
                                                         SS.SS_mbd,SS_new.SS_mbd,SS.SS_mbp,SS.SS_mbp,mm);

            total_Posterior_new = Compute_Total_Posterior(Sum_Num_new,Posterior_new,M);
            if(total_Posterior_new > total_Posterior)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                Sum_Num = Sum_Num_new;
                SS.SS_md[mm] = SS_new.SS_md[mm];
                SS.SS_mbd[mm] = SS_new.SS_mbd[mm];
            }
            else
            {
                if(total_Posterior_new - total_Posterior > -500.0)
                    ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
                else
                    ratio = 0.0;
                rand_t = gsl_rng_uniform (r);
                //rand_t = rand_number_collection.front();
                //rand_number_collection.pop();
                if(rand_t <= ratio)
                {
                    total_Posterior = total_Posterior_new;

                    Posterior = Posterior_new;
                    Sum_Num = Sum_Num_new;
                    SS.SS_md[mm] = SS_new.SS_md[mm];
                    SS.SS_mbd[mm] = SS_new.SS_mbd[mm];
                }
                else
                {
                    Module_Center.Beta_mid[i*(M+1) + mm] = pre_module_center;
                    //SS_new.SS_md[mm] = SS.SS_md[mm];
                    //SS_new.SS_mbd[mm] = SS.SS_mbd[mm];
                }
            }
        }
    }
    free(SS_new.SS_mbd);
    //free(SS_new.SS_mbp);
    free(SS_new.SS_md);
    //free(SS_new.SS_mp);
    //gsl_rng_free(r);
}

void Change_Phenotype_Module_Center(gsl_matrix* Matrix1,gsl_matrix* Matrix2,\
                                    struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                                    int* num_G, gsl_rng* r)
{
    int i;
    int M = Module_Count.M;

    struct_SS SS_new;
    struct_Sum_Num Sum_Num_new;
    struct_Posterior Posterior_new;
    //SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    int mmi;
    /*for(mmi = 0;mmi<=M;mmi++)
    {
        SS_new.SS_md[mmi] = SS.SS_md[mmi];
        SS_new.SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new.SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new.SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;
    double pre_module_center;

    for(int mm=1; mm<=M; mm++)
    {
        if(num_G[mm] == 0)
            continue;
        for(i = 0; i<numP; i++)
        {
            if(Indicator.Imp[i*(M+1)+mm] != 1)
                continue;
            Sum_Num_new = Sum_Num;
            Posterior_new = Posterior;

            rand_t = gsl_rng_uniform (r);
            //rand_t = rand_number_collection.front();
            //rand_number_collection.pop();
            pre_module_center = Module_Center.Beta_mip[i*(M+1) + mm];
            if(rand_t<= 0.5)
                Module_Center.Beta_mip[i*(M+1) + mm] += Delta_Beta;
            else
                Module_Center.Beta_mip[i*(M+1) + mm] -= Delta_Beta;

            //======================================================
            change_SS_mp(Matrix2,Indicator.Ig,Indicator.Imp,Module_Center.Beta_mip,SS_new.SS_mp,mm);
            change_SS_mbp(Indicator.Imp,Module_Center.Beta_mip,SS_new.SS_mbp,mm);

            Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Indicator.Ig,\
                                                         SS.SS_md,SS.SS_md,SS.SS_mp,SS_new.SS_mp,mm);
            Posterior_new.part3 = change_Posterior_part3(Posterior_new.part3,Indicator.Imd,Indicator.Imd,Indicator.Imp,Indicator.Imp,\
                                                         SS.SS_mbd,SS.SS_mbd,SS.SS_mbp,SS_new.SS_mbp,mm);

            total_Posterior_new = Compute_Total_Posterior(Sum_Num_new,Posterior_new,M);
            if(total_Posterior_new > total_Posterior)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                Sum_Num = Sum_Num_new;

                SS.SS_mp[mm] = SS_new.SS_mp[mm];
                SS.SS_mbp[mm] = SS_new.SS_mbp[mm];
            }
            else
            {
                if(total_Posterior_new - total_Posterior > -500.0)
                    ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
                else
                    ratio = 0.0;
                rand_t = gsl_rng_uniform (r);
                //rand_t = rand_number_collection.front();
                //rand_number_collection.pop();
                if(rand_t <= ratio)
                {
                    total_Posterior = total_Posterior_new;

                    Posterior = Posterior_new;
                    Sum_Num = Sum_Num_new;
                    SS.SS_mp[mm] = SS_new.SS_mp[mm];
                    SS.SS_mbp[mm] = SS_new.SS_mbp[mm];
                }
                else
                {
                    Module_Center.Beta_mip[i*(M+1) + mm] = pre_module_center;
                    //SS_new.SS_mp[mm] = SS.SS_mp[mm];
                    //SS_new.SS_mbp[mm] = SS.SS_mbp[mm];
                }
            }
        }
    }
    //free(SS_new.SS_mbd);
    free(SS_new.SS_mbp);
    //free(SS_new.SS_md);
    free(SS_new.SS_mp);
    //gsl_rng_free(r);
}
//====================================================
void Exchange_Drug_Indicator(gsl_matrix* Matrix1, gsl_matrix* Matrix2, \
                             struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                             int* num_G, gsl_rng* r)
{
    int i;
    int M = Module_Count.M;

    int num_md0 = 0;
    int num_md1 = 0;
    int index_md0[numD];
    int index_md1[numD];
    int select_0,select_1;

    int mmi,mm;

    struct_SS SS_new;
    struct_Posterior Posterior_new;

    SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    /*for(mmi = 0;mmi<=M;mmi++)
    {
        SS_new.SS_md[mmi] = SS.SS_md[mmi];
        SS_new.SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new.SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new.SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;

    int* Imd_new = (int*)malloc(sizeof(int)*numD*(M+1));
    for(i = 0; i<numD*(M+1);i++)
    {
        Imd_new[i] = Indicator.Imd[i];
    }

    for(mm = 1; mm<=M; mm++)
    {
        if(num_G[mm] == 0)
            continue;
        num_md0 = 0;
        num_md1 = 0;
        for(i = 0; i<numD; i++)
        {
            if(Indicator.Imd[i*(M+1)+mm] == 1)
                index_md1[num_md1++] = i;
            else
                index_md0[num_md0++] = i;
        }
        if(num_md0 ==0 || num_md1 == 0)
            continue;
        gsl_ran_sample(r,&select_0,1,index_md0,num_md0,sizeof(int));
        gsl_ran_sample(r,&select_1,1,index_md1,num_md1,sizeof(int));
        //select_0 = index_md0[(int)floor(num_md0*1.0/2)];
        //select_1 = index_md1[(int)floor(num_md1*1.0/2)];

        if(Imd_new[select_0*(M+1)+mm] != 0 || Imd_new[select_1*(M+1)+mm] != 1)
            cout<<"error"<<endl;

        Imd_new[select_0*(M+1)+mm] = 1;
        Imd_new[select_1*(M+1)+mm] = 0;

        Posterior_new = Posterior;
        //=======================================
        change_SS_md(Matrix1,Indicator.Ig,Imd_new,Module_Center.Beta_mid,SS_new.SS_md,mm);
        change_SS_mbd(Imd_new,Module_Center.Beta_mid,SS_new.SS_mbd,mm);

        Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Indicator.Ig,\
                                                     SS.SS_md,SS_new.SS_md,SS.SS_mp,SS.SS_mp,mm);
        Posterior_new.part3 = change_Posterior_part3(Posterior_new.part3,Indicator.Imd,Imd_new,Indicator.Imp,Indicator.Imp,\
                                                     SS.SS_mbd,SS_new.SS_mbd,SS.SS_mbp,SS.SS_mbp,mm);

        total_Posterior_new = Compute_Total_Posterior(Sum_Num,Posterior_new,M);
        //=============================================
        if(total_Posterior_new > total_Posterior)
        {
            total_Posterior = total_Posterior_new;

            Posterior = Posterior_new;
            SS.SS_md[mm] = SS_new.SS_md[mm];
            SS.SS_mbd[mm] = SS_new.SS_mbd[mm];

            Indicator.Imd[select_0*(M+1)+mm] = 1;
            Indicator.Imd[select_1*(M+1)+mm] = 0;
        }
        else
        {
            if(total_Posterior_new - total_Posterior > -500.0)
                ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
            else
                ratio = 0.0;
            rand_t = gsl_rng_uniform (r);
            //rand_t = rand_number_collection.front();
            //rand_number_collection.pop();
            if(rand_t <= ratio)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                SS.SS_md[mm] = SS_new.SS_md[mm];
                SS.SS_mbd[mm] = SS_new.SS_mbd[mm];

                Indicator.Imd[select_0*(M+1)+mm] = 1;
                Indicator.Imd[select_1*(M+1)+mm] = 0;
            }
            else
            {
                //SS_new.SS_md[mm] = SS.SS_md[mm];
                //SS_new.SS_mbd[mm] = SS.SS_mbd[mm];
                Imd_new[select_0*(M+1)+mm] = 0;
                Imd_new[select_1*(M+1)+mm] = 1;
            }
        }
    }
    free(SS_new.SS_mbd);
    //free(SS_new.SS_mbp);
    free(SS_new.SS_md);
    //free(SS_new.SS_mp);
    free(Imd_new);
    //gsl_rng_free(r);
}

void Exchange_Phenotype_Indicator(gsl_matrix* Matrix1, gsl_matrix* Matrix2, \
                                  struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                                  int* num_G, gsl_rng* r)
{
    int i;
    int M = Module_Count.M;

    int num_mp0 = 0;
    int num_mp1 = 0;
    int index_mp0[numP];
    int index_mp1[numP];
    int select_0,select_1;
    int mmi,mm;

    struct_SS SS_new;
    struct_Posterior Posterior_new;

    //SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    /*for(mmi = 0;mmi<=M;mmi++)
    {
        SS_new.SS_md[mmi] = SS.SS_md[mmi];
        SS_new.SS_mp[mmi] = SS.SS_mp[mmi];
        SS_new.SS_mbd[mmi] = SS.SS_mbd[mmi];
        SS_new.SS_mbp[mmi] = SS.SS_mbp[mmi];
    }*/

    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;

    //int Imp_new[numP*(M+1)];
    int* Imp_new = (int*)malloc(sizeof(int)*numP*(M+1));

    for(i = 0; i<numP*(M+1);i++)
    {
        Imp_new[i] = Indicator.Imp[i];
    }

    for(mm = 1; mm<=M; mm++)
    {
        if(num_G[mm] == 0)
            continue;

        num_mp0 = 0;
        num_mp1 = 0;
        for(i = 0; i<numP; i++)
        {
            if(Indicator.Imp[i*(M+1)+mm] == 1)
                index_mp1[num_mp1++] = i;
            else
                index_mp0[num_mp0++] = i;
        }
        if(num_mp0 ==0 || num_mp1 == 0)
            continue;
        gsl_ran_sample(r,&select_0,1,index_mp0,num_mp0,sizeof(int));
        gsl_ran_sample(r,&select_1,1,index_mp1,num_mp1,sizeof(int));

        //select_0 = index_mp0[(int)floor(num_mp0*1.0/2)];
        //select_1 = index_mp1[(int)floor(num_mp1*1.0/2)];
        if(Imp_new[select_0*(M+1)+mm] != 0 || Imp_new[select_1*(M+1)+mm] != 1)
            cout<<"error"<<endl;

        Imp_new[select_0*(M+1)+mm] = 1;
        Imp_new[select_1*(M+1)+mm] = 0;

        Posterior_new = Posterior;
        //=======================================
        change_SS_mp(Matrix2,Indicator.Ig,Imp_new,Module_Center.Beta_mip,SS_new.SS_mp,mm);
        change_SS_mbp(Imp_new,Module_Center.Beta_mip,SS_new.SS_mbp,mm);

        Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Indicator.Ig,\
                                                     SS.SS_md,SS.SS_md,SS.SS_mp,SS_new.SS_mp,mm);
        Posterior_new.part3 = change_Posterior_part3(Posterior_new.part3,Indicator.Imd,Indicator.Imd,Indicator.Imp,Imp_new,\
                                                     SS.SS_mbd,SS.SS_mbd,SS.SS_mbp,SS_new.SS_mbp,mm);

        total_Posterior_new = Compute_Total_Posterior(Sum_Num,Posterior_new,M);
        //=============================================
        if(total_Posterior_new > total_Posterior)
        {
            total_Posterior = total_Posterior_new;

            Posterior = Posterior_new;
            SS.SS_mp[mm] = SS_new.SS_mp[mm];
            SS.SS_mbp[mm] = SS_new.SS_mbp[mm];

            Indicator.Imp[select_0*(M+1)+mm] = 1;
            Indicator.Imp[select_1*(M+1)+mm] = 0;
        }
        else
        {
            if(total_Posterior_new - total_Posterior > -500.0)
                ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
            else
                ratio = 0.0;
            rand_t = gsl_rng_uniform (r);
            //rand_t = rand_number_collection.front();
            //rand_number_collection.pop();
            if(rand_t <= ratio)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                SS.SS_mp[mm] = SS_new.SS_mp[mm];
                SS.SS_mbp[mm] = SS_new.SS_mbp[mm];

                Indicator.Imp[select_0*(M+1)+mm] = 1;
                Indicator.Imp[select_1*(M+1)+mm] = 0;
            }
            else
            {
                //SS_new.SS_mp[mm] = SS.SS_mp[mm];
                //SS_new.SS_mbp[mm] = SS.SS_mbp[mm];
                Imp_new[select_0*(M+1)+mm] = 0;
                Imp_new[select_1*(M+1)+mm] = 1;
            }
        }
    }
    //free(SS_new.SS_mbd);
    free(SS_new.SS_mbp);
    //free(SS_new.SS_md);
    free(SS_new.SS_mp);
    free(Imp_new);
    //gsl_rng_free(r);
}

void Exchange_Gene_Indicator_Module_NULL(gsl_matrix* Matrix1, gsl_matrix* Matrix2, \
                                         struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                                         gsl_rng* r)
{
    //gsl_rng * r;
    //timeb this_time;
    //ftime(&this_time);
    //gsl_rng_default_seed = (unsigned long)(this_time.millitm)*getpid();
    //r = gsl_rng_alloc(gsl_rng_ranlxd1);

    int M = Module_Count.M;
    //---------------------------------------------
    int num_G[M+1];
    //int index_G[(M+1)*numG];
    int* index_G = (int*)malloc((M+1)*numG*sizeof(int));
    int Ig_new[numG];
    for(int i = 0; i<numG; i++)
        Ig_new[i] = Indicator.Ig[i];

    struct_SS SS_new;
    struct_Posterior Posterior_new;
    SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    /*for(int mm = 0;mm<=M;mm++)
    {
        SS_new.SS_md[mm] = SS.SS_md[mm];
        SS_new.SS_mp[mm] = SS.SS_mp[mm];
        SS_new.SS_mbd[mm] = SS.SS_mbd[mm];
        SS_new.SS_mbp[mm] = SS.SS_mbp[mm];
    }*/
    //---------------------------------------------
    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;

    for(int mm=1; mm<=M;mm++)
    {

        for(int i = 0; i<= M; i++)
            num_G[i] = 0;
        for(int i = 0; i<numG; i++)
            index_G[Indicator.Ig[i]*numG + num_G[Indicator.Ig[i]]++] = i;
        int select_m,select_0;
        if(num_G[mm] == 0 || num_G[0] == 0)
            continue;
        //---------------------------------------------
        Posterior_new = Posterior;
        gsl_ran_sample(r,&select_m,1,&index_G[mm*numG],num_G[mm],sizeof(int));
        gsl_ran_sample(r,&select_0,1,&index_G[0],num_G[0],sizeof(int));

        //select_0 = index_G[(int)floor(num_G[0]*1.0/2)];
        //select_m = index_G[mm*numG + (int)floor(num_G[mm]*1.0/2)];
        if(Ig_new[select_m] != mm || Ig_new[select_0] != 0)
            cout<<"error"<<endl;

        Ig_new[select_m] = 0;
        Ig_new[select_0] = mm;
        //---------------------------------------
        change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new.SS_md,mm);
        change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new.SS_mp,mm);

        Posterior_new.part1 = change_Posterior_part1(Matrix1,Matrix2,Ig_new);
        Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Ig_new,\
                                                     SS.SS_md,SS_new.SS_md,SS.SS_mp,SS_new.SS_mp,mm);
        total_Posterior_new = Compute_Total_Posterior(Sum_Num,Posterior_new,M);
        //---------------------------------------
        if(total_Posterior_new > total_Posterior)
        {
            total_Posterior = total_Posterior_new;

            Posterior = Posterior_new;
            SS.SS_md[mm] = SS_new.SS_md[mm];
            SS.SS_mp[mm] = SS_new.SS_mp[mm];
            Indicator.Ig[select_m] = 0;
            Indicator.Ig[select_0] = mm;
        }
        else
        {
            if(total_Posterior_new - total_Posterior > -500.0)
                ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
            else
                ratio = 0.0;
            rand_t = gsl_rng_uniform (r);
            //rand_t = rand_number_collection.front();
            //rand_number_collection.pop();
            if(rand_t <= ratio)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                SS.SS_md[mm] = SS_new.SS_md[mm];
                SS.SS_mp[mm] = SS_new.SS_mp[mm];
                Indicator.Ig[select_m] = 0;
                Indicator.Ig[select_0] = mm;
            }
            else
            {
                SS_new.SS_md[mm] = SS.SS_md[mm];
                SS_new.SS_mp[mm] = SS.SS_mp[mm];
                Ig_new[select_m] = mm;
                Ig_new[select_0] = 0;
            }
        }
    }
    //free(SS_new.SS_mbd);
    //free(SS_new.SS_mbp);
    free(SS_new.SS_md);
    free(SS_new.SS_mp);
    free(index_G);
    //gsl_rng_free(r);
}

void Exchange_Gene_Indicator_Module_Module(gsl_matrix* Matrix1, gsl_matrix* Matrix2, \
                                           struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                                           gsl_rng* r)
{
    //gsl_rng * r;
    //timeb this_time;
    //ftime(&this_time);
    //gsl_rng_default_seed = (unsigned long)(this_time.millitm)*getpid();
    //r = gsl_rng_alloc(gsl_rng_ranlxd1);

    int M = Module_Count.M;
    int mmi,mmj,i;
    int num_G[M+1];
    int* index_G = (int*)malloc((M+1)*numG*sizeof(int));
    int Ig_new[numG];
    for(i = 0; i<numG; i++)
        Ig_new[i] = Indicator.Ig[i];
    struct_SS SS_new;
    struct_Posterior Posterior_new;
    struct_Sum_Num Sum_Num_new;

    SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    //SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    /*for(int mm = 0;mm<=M;mm++)
    {
        SS_new.SS_md[mm] = SS.SS_md[mm];
        SS_new.SS_mp[mm] = SS.SS_mp[mm];
        SS_new.SS_mbd[mm] = SS.SS_mbd[mm];
        SS_new.SS_mbp[mm] = SS.SS_mbp[mm];
    }*/
    //---------------------------------------------
    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;

    int select_mmi,select_mmj;
    //---------------------------------------------
    for(mmi = 1; mmi<=M; mmi++)
    {
        for(mmj = mmi +1; mmj<= M; mmj++)
        {

            for(i = 0; i<= M; i++)
                num_G[i] = 0;
            for(i = 0; i<numG; i++)
                index_G[Indicator.Ig[i]*numG + num_G[Indicator.Ig[i]]++] = i;

            if(num_G[mmi] == 0||num_G[mmj] == 0)
                continue;
            //---------------------------------------------
            Posterior_new = Posterior;
            gsl_ran_sample(r,&select_mmi,1,&index_G[mmi*numG],num_G[mmi],sizeof(int));
            gsl_ran_sample(r,&select_mmj,1,&index_G[mmj*numG],num_G[mmj],sizeof(int));

            //select_mmi = index_G[mmi*numG + (int)floor(num_G[mmi]*1.0/2)];
            //select_mmj = index_G[mmj*numG + (int)floor(num_G[mmj]*1.0/2)];
            if(Indicator.Ig[select_mmi] != mmi || Indicator.Ig[select_mmj] != mmj)
                cout<<"error"<<endl;
            Ig_new[select_mmi] = mmj;
            Ig_new[select_mmj] = mmi;
            //---------------------------------------
            change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new.SS_md,mmi);
            change_SS_md(Matrix1,Ig_new,Indicator.Imd,Module_Center.Beta_mid,SS_new.SS_md,mmj);
            change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new.SS_mp,mmi);
            change_SS_mp(Matrix2,Ig_new,Indicator.Imp,Module_Center.Beta_mip,SS_new.SS_mp,mmj);

            Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Ig_new,\
                                                         SS.SS_md,SS_new.SS_md,SS.SS_mp,SS_new.SS_mp,mmi);
            Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Ig_new,\
                                                         SS.SS_md,SS_new.SS_md,SS.SS_mp,SS_new.SS_mp,mmj);
            total_Posterior_new = Compute_Total_Posterior(Sum_Num,Posterior_new,M);
            //---------------------------------------
            if(total_Posterior_new > total_Posterior)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                SS.SS_md[mmi] = SS_new.SS_md[mmi];
                SS.SS_md[mmj] = SS_new.SS_md[mmj];
                SS.SS_mp[mmi] = SS_new.SS_mp[mmi];
                SS.SS_mp[mmj] = SS_new.SS_mp[mmj];
                Indicator.Ig[select_mmi] = mmj;
                Indicator.Ig[select_mmj] = mmi;
            }
            else
            {
                if(total_Posterior_new - total_Posterior > -500.0)
                    ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
                else
                    ratio = 0.0;
                rand_t = gsl_rng_uniform (r);
                //rand_t = rand_number_collection.front();
                //rand_number_collection.pop();
                if(rand_t <= ratio)
                {
                    total_Posterior = total_Posterior_new;

                    Posterior = Posterior_new;
                    SS.SS_md[mmi] = SS_new.SS_md[mmi];
                    SS.SS_md[mmj] = SS_new.SS_md[mmj];
                    SS.SS_mp[mmi] = SS_new.SS_mp[mmi];
                    SS.SS_mp[mmj] = SS_new.SS_mp[mmj];
                    Indicator.Ig[select_mmi] = mmj;
                    Indicator.Ig[select_mmj] = mmi;
                }
                else
                {
                    SS_new.SS_md[mmi] = SS.SS_md[mmi];
                    SS_new.SS_md[mmj] = SS.SS_md[mmj];
                    SS_new.SS_mp[mmi] = SS.SS_mp[mmi];
                    SS_new.SS_mp[mmj] = SS.SS_mp[mmj];
                    Ig_new[select_mmi] = mmi;
                    Ig_new[select_mmj] = mmj;
                }
            }
        }
    }

    //free(SS_new.SS_mbd);
    //free(SS_new.SS_mbp);
    free(SS_new.SS_md);
    free(SS_new.SS_mp);
    free(index_G);
    //gsl_rng_free(r);
}

void Exchange_Module_to_NULL(gsl_matrix* Matrix1, gsl_matrix* Matrix2, \
                             struct_Indicator& Indicator,struct_Module_Center& Module_Center,\
                             gsl_rng* r)
{
    int M = Module_Count.M;
    register int i;

    double* Beta_mid_new = (double*)malloc(numD*(M+1)*sizeof(double));
    double* Beta_mip_new = (double*)malloc(numP*(M+1)*sizeof(double));

    int* Imd_new = (int*)malloc(numD*(M+1)*sizeof(int));
    int* Imp_new = (int*)malloc(numP*(M+1)*sizeof(int));
    int* Ig_new = (int*)malloc(numG*sizeof(int));

    for(i = 0;i<numD*(M+1); i++)
    {
        Beta_mid_new[i] = Module_Center.Beta_mid[i];
        Imd_new[i] = Indicator.Imd[i];
    }
    for(i = 0; i<numP*(M+1); i++)
    {
        Beta_mip_new[i] = Module_Center.Beta_mip[i];
        Imp_new[i] = Indicator.Imp[i];
    }
    for(i = 0; i<numG; i++)
        Ig_new[i] = Indicator.Ig[i];

    struct_SS SS_new;
    struct_Posterior Posterior_new;
    struct_Sum_Num Sum_Num_new;

    SS_new.SS_md = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mp = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbd = (double*)malloc((M+1)*sizeof(double));
    SS_new.SS_mbp = (double*)malloc((M+1)*sizeof(double));
    /*for(int mm = 0;mm<=M;mm++)
    {
        SS_new.SS_md[mm] = SS.SS_md[mm];
        SS_new.SS_mp[mm] = SS.SS_mp[mm];
        SS_new.SS_mbd[mm] = SS.SS_mbd[mm];
        SS_new.SS_mbp[mm] = SS.SS_mbp[mm];
    }*/

    double total_Posterior = Compute_Total_Posterior(Sum_Num,Posterior,M);
    double total_Posterior_new;
    double rand_t;
    double ratio;

    int num_G[M+1];

    //gsl_rng * r;
    //timeb this_time;
    //ftime(&this_time);
    //gsl_rng_default_seed = (unsigned long)(this_time.millitm)*getpid();
    //r = gsl_rng_alloc(gsl_rng_ranlxd1);

    for(int mm=1;mm<=M;mm++)
    {

        for(i = 0; i<= M; i++)
            num_G[i] = 0;
        for(i = 0; i<numG; i++)
            num_G[Indicator.Ig[i]]++;

        if(num_G[mm] == 0)
            continue;
        for(i=0; i<numD; i++)
        {
            Beta_mid_new[i*(M+1) + mm] = Module_Center.Beta_mid[i*(M+1)];
            Beta_mid_new[i*(M+1)] = Module_Center.Beta_mid[i*(M+1) + mm];
            //Imd_new[i*(M+1) + mm] = Indicator.Imd[i*(M+1)];
            //Imd_new[i*(M+1)] = Indicator.Imd[i*(M+1) + mm];
        }
        for(i=0; i<numP; i++)
        {
            Beta_mip_new[i*(M+1) + mm] = Module_Center.Beta_mip[i*(M+1)];
            Beta_mip_new[i*(M+1)] = Module_Center.Beta_mip[i*(M+1)+mm];
            //Imp_new[i*(M+1) + mm] = Indicator.Imp[i*(M+1)];
            //Imp_new[i*(M+1)] = Indicator.Imp[i*(M+1) + mm];
        }
        for(i = 0; i< numG; i++)
        {
            if(Indicator.Ig[i] == mm)
                Ig_new[i] =0;
            if(Indicator.Ig[i] == 0)
                Ig_new[i] = mm;
        }

        Posterior_new = Posterior;
        //==============================
        change_SS_md(Matrix1,Ig_new,Imd_new,Beta_mid_new,SS_new.SS_md,mm);
        change_SS_mp(Matrix2,Ig_new,Imp_new,Beta_mip_new,SS_new.SS_mp,mm);
        change_SS_mbd(Imd_new,Beta_mid_new,SS_new.SS_mbd,mm);
        change_SS_mbp(Imp_new,Beta_mip_new,SS_new.SS_mbp,mm);
        //==============================
        Posterior_new.part1 = change_Posterior_part1(Matrix1,Matrix2,Ig_new);
        Posterior_new.part2 = change_Posterior_part2(Matrix1,Matrix2,Posterior_new.part2,Indicator.Ig,Ig_new,\
                                                     SS.SS_md,SS_new.SS_md,SS.SS_mp,SS_new.SS_mp,mm);
        Posterior_new.part3 = change_Posterior_part3(Posterior_new.part3,Indicator.Imd,Imd_new,Indicator.Imp,Imp_new,\
                                                     SS.SS_mbd,SS_new.SS_mbd,SS.SS_mbp,SS_new.SS_mbp,mm);
        compute_Sum_Num_local(Ig_new,Imd_new,Imp_new,Sum_Num_new);
        total_Posterior_new = Compute_Total_Posterior(Sum_Num_new,Posterior_new,M);
        //total_Posterior_new = 10000000000.0;
        //==============================
        if(total_Posterior_new > total_Posterior)
        {
            total_Posterior = total_Posterior_new;

            Posterior = Posterior_new;
            Sum_Num = Sum_Num_new;

            SS.SS_md[mm] = SS_new.SS_md[mm];
            SS.SS_mp[mm] = SS_new.SS_mp[mm];
            SS.SS_mbd[mm] = SS_new.SS_mbd[mm];
            SS.SS_mbp[mm] = SS_new.SS_mbp[mm];

            for(i=0; i<numD; i++)
            {
                Module_Center.Beta_mid[i*(M+1) + mm] = Beta_mid_new[i*(M+1) + mm];
                Module_Center.Beta_mid[i*(M+1)] = Beta_mid_new[i*(M+1)];
                //Indicator.Imd[i*(M+1) + mm] = Imd_new[i*(M+1) + mm];
                //Indicator.Imd[i*(M+1)] = Imd_new[i*(M+1)];
            }
            for(i=0; i<numP; i++)
            {
                Module_Center.Beta_mip[i*(M+1) + mm] = Beta_mip_new[i*(M+1) + mm];
                Module_Center.Beta_mip[i*(M+1)] = Beta_mip_new[i*(M+1)];
                //Indicator.Imp[i*(M+1) + mm] = Imp_new[i*(M+1) + mm];
                //Indicator.Imp[i*(M+1)] = Imp_new[i*(M+1)];
            }
            for(i = 0; i< numG; i++)
                Indicator.Ig[i] = Ig_new[i];
        }
        else
        {
            if(total_Posterior_new - total_Posterior > -500.0)
                ratio = gsl_sf_exp(total_Posterior_new - total_Posterior);
            else
                ratio = 0.0;
            rand_t = gsl_rng_uniform (r);
            //rand_t = rand_number_collection.front();
            //rand_number_collection.pop();
            if(rand_t <= ratio)
            {
                total_Posterior = total_Posterior_new;

                Posterior = Posterior_new;
                Sum_Num = Sum_Num_new;

                SS.SS_md[mm] = SS_new.SS_md[mm];
                SS.SS_mp[mm] = SS_new.SS_mp[mm];
                SS.SS_mbd[mm] = SS_new.SS_mbd[mm];
                SS.SS_mbp[mm] = SS_new.SS_mbp[mm];

                for(i=0; i<numD; i++)
                {
                    Module_Center.Beta_mid[i*(M+1) + mm] = Beta_mid_new[i*(M+1) + mm];
                    Module_Center.Beta_mid[i*(M+1)] = Beta_mid_new[i*(M+1)];
                    //Indicator.Imd[i*(M+1) + mm] = Imd_new[i*(M+1) + mm];
                    //Indicator.Imd[i*(M+1)] = Imd_new[i*(M+1)];
                }
                for(i=0; i<numP; i++)
                {
                    Module_Center.Beta_mip[i*(M+1) + mm] = Beta_mip_new[i*(M+1) + mm];
                    Module_Center.Beta_mip[i*(M+1)] = Beta_mip_new[i*(M+1)];
                    //Indicator.Imp[i*(M+1) + mm] = Imp_new[i*(M+1) + mm];
                    //Indicator.Imp[i*(M+1)] = Imp_new[i*(M+1)];
                }
                for(i = 0; i< numG; i++)
                    Indicator.Ig[i] = Ig_new[i];
            }
            else
            {
                SS_new.SS_md[mm] = SS.SS_md[mm];
                SS_new.SS_mbd[mm] = SS.SS_mbd[mm];
                SS_new.SS_mp[mm] = SS.SS_mp[mm];
                SS_new.SS_mbp[mm] = SS.SS_mbp[mm];
                for(i=0; i<numD; i++)
                {
                    Beta_mid_new[i*(M+1)] = Module_Center.Beta_mid[i*(M+1)];
                    Beta_mid_new[i*(M+1)+mm] = Module_Center.Beta_mid[i*(M+1) + mm];
                    //Imd_new[i*(M+1)] = Indicator.Imd[i*(M+1)];
                    //Imd_new[i*(M+1)+mm] = Indicator.Imd[i*(M+1) + mm];
                }
                for(i=0; i<numP; i++)
                {
                    Beta_mip_new[i*(M+1)] = Module_Center.Beta_mip[i*(M+1)];
                    Beta_mip_new[i*(M+1)+mm] = Module_Center.Beta_mip[i*(M+1)+mm];
                    //Imp_new[i*(M+1)] = Indicator.Imp[i*(M+1)];
                    //Imp_new[i*(M+1)+mm] = Indicator.Imp[i*(M+1) + mm];
                }
                for(i = 0; i< numG; i++)
                    Ig_new[i] = Indicator.Ig[i];
            }
        }

    }
    free(SS_new.SS_mbd);
    free(SS_new.SS_mbp);
    free(SS_new.SS_md);
    free(SS_new.SS_mp);

    free(Imd_new);
    free(Imp_new);
    free(Ig_new);
    free(Beta_mid_new);
    free(Beta_mip_new);

    //gsl_rng_free(r);
}
//==================================================


//==================================================
void write_Ig_to_file_line(int* Ig,char* file_name)
{

    ofstream output(file_name,ios::app);
    for(int i = 0; i<numG; i++)
    {
        output<<Ig[i]<<"\t";
    }
    output<<endl;
    output.close();
}

void write_Ig_probability_to_file(double* Ig_probability,char* file_name,int M)
{
    ofstream output(file_name);
    for (int mm = 0; mm<= M; mm++)
    {
        for(int i = 0; i<numG; i++)
        {
            output<<Ig_probability[mm*numG + i]<<"\t";
        }
    output<<endl;
    }

    output.close();
}

void write_Imd_to_file_matrix(int* Imd,char* file_name, int M)
{
    ofstream output(file_name,ios::app);
    for(int i = 0; i<numD; i++)
    {
        for (int mm = 0; mm<=M;mm++)
        {
            output<<Imd[i*(M+1)+mm]<<"\t";
        }
        output<<endl;
    }
    output.close();
}

void write_Imd_to_file_line(int* Imd,char* file_name, int M)
{
    ofstream output(file_name,ios::app);
    for (int mm = 0; mm<= M; mm++ )
    {
        for(int i = 0; i<numD; i++)
        {
            output<<Imd[(M+1)*i + mm]<<"\t";
        }
    }
    output<<endl;

    output.close();
}

void write_Imd_probability_to_file(double* Imd_probability,char* file_name, int M)
{
    ofstream output(file_name);
    for (int mm = 1; mm<= M; mm++)
    {
        for(int i = 0; i<numD; i++)
        {
            output<<Imd_probability[(mm-1)*numD + i]<<"\t";
        }
        output<<endl;
    }

    output.close();
}


void write_Imp_to_file_matrix(int* Imp, char* file_name, int M)
{
    ofstream output(file_name);
    for(int i = 0; i<numP; i++)
    {
        for (int mm = 0; mm<=M;mm++)
        {
            output<<Imp[i*(M+1)+mm]<<"\t";
        }
        output<<endl;
    }
    output.close();
}

void write_Imp_to_file_line(int* Imp,char* file_name, int M)
{
    ofstream output(file_name,ios::app);
    for (int mm = 0; mm<= M; mm++)
    {
        for(int i = 0; i<numP; i++ )
        {
            output<<Imp[i*(M+1) + mm]<<"\t";
        }
    }
    output<<endl;
    output.close();
}

void write_Imp_probability_to_file(double* Imp_probability,char* file_name, int M)
{
    ofstream output(file_name);
    for (int mm = 1; mm<= M; mm++)
    {
        for(int i = 0; i<numP; i++)
        {
            output<<Imp_probability[(mm-1)*numP + i]<<"\t";
        }
    output<<endl;
    }

    output.close();
}


void write_SS_to_file(struct_SS& SS_local)
{
    ofstream output("result_SS.txt");
    int M = Module_Count.M;
    for(int i = 0; i<=M; i++)
    {
        output<<SS_local.SS_md[i]<<"\t";
    }
    output<<endl;
    for(int i = 0; i<=M; i++)
    {
        output<<SS_local.SS_mp[i]<<"\t";
    }
    output<<endl;
    for(int i = 0; i<=M; i++)
    {
        output<<SS_local.SS_mbd[i]<<"\t";
    }
    output<<endl;
    for(int i = 0; i<=M; i++)
    {
        output<<SS_local.SS_mbp[i]<<"\t";
    }
    output<<endl;
    output.close();
}

void write_Posterior_to_file(struct_Posterior Posterior_local)
{
    ofstream output("result_Posterior.txt");
    output<<Posterior_local.part1<<endl;
    output<<Posterior_local.part2<<endl;
    output<<Posterior_local.part3<<endl;
    output.close();
}

void write_Beta_mid_to_file_line(double* Beta_mid,char* file_name, int M)
{
    ofstream output(file_name,ios::app);
    for(int i = 0; i<numD; i++)
    {
        for (int mm = 0; mm<=M;mm++)
        {
            output<<Beta_mid[i*(M+1)+mm]<<"\t";
        }
    }
    output<<endl;
    output.close();
}

void write_Beta_mip_to_file_line(double* Beta_mip,char* file_name, int M)
{
    ofstream output(file_name,ios::app);
    for(int i = 0; i<numP; i++)
    {
        for (int mm = 0; mm<=M;mm++)
        {
            output<<Beta_mip[i*(M+1)+mm]<<"\t";
        }
    }
    output<<endl;
    output.close();
}

void write_Process_Batch_to_file(struct_Indicator *Indicator_Process, struct_Module_Center *Module_Center_Process,\
                                 int Write_Batch, int M,\
                                 char *output_file_Ig_Process, char *output_file_Imd_Process, char *output_file_Imp_Process,\
                                 char *output_file_Beta_mid_Process, char *output_file_Beta_mip_Process)
{
    int i,j,mm;
    ofstream output_Ig(output_file_Ig_Process,ios::app);
    ofstream output_Imd(output_file_Imd_Process,ios::app);
    ofstream output_Imp(output_file_Imp_Process,ios::app);

    ofstream output_Beta_mid(output_file_Beta_mid_Process,ios::app);
    ofstream output_Beta_mip(output_file_Beta_mip_Process,ios::app);

    if(output_Beta_mid == NULL || output_Beta_mip == NULL ||\
        output_Ig == NULL || output_Imd == NULL || output_Imp == NULL )
    {
        cout<<"Error: There is a problem in creating files that record the interation process."<<endl;
        exit(1);
    }

    for(i = 0; i < Write_Batch; i++)
    {
        for(j = 0; j<numG; j++)
            output_Ig<<Indicator_Process[i].Ig[j]<<"\t";
        output_Ig<<endl;

        for (mm = 0; mm<= M; mm++ )
        {
            for( j = 0; j<numD; j++)
            {
                output_Imd<<Indicator_Process[i].Imd[(M+1)*j + mm]<<"\t";
                output_Beta_mid<<Module_Center_Process[i].Beta_mid[(M+1)*j + mm]<<"\t";
            }
        }
        output_Imd<<endl;
        output_Beta_mid<<endl;

        for (mm = 0; mm<= M; mm++)
        {
            for(j = 0; j<numP; j++ )
            {
                output_Imp<<Indicator_Process[i].Imp[(M+1)*j + mm]<<"\t";
                output_Beta_mip<<Module_Center_Process[i].Beta_mip[(M+1)*j + mm]<<"\t";
            }
        }
        output_Imp<<endl;
        output_Beta_mip<<endl;

    }
    output_Ig.close();
    output_Imd.close();
    output_Imp.close();
    output_Beta_mid.close();
    output_Beta_mip.close();

}



//==================================================

void read_configure_file(int &M, int &Burn_In_Iteration, int &Sample_Stride, int &Sample_Num,\
                         char *&input_file_name1, char *&input_file_name2,\
                         char *&output_file_name1, char *&output_file_name2, char *&output_file_name3,\
                         char *&output_file_Ig_Process, char *&output_file_Imd_Process, char *&output_file_Imp_Process,\
                         char *&output_file_Beta_mid_Process, char *&output_file_Beta_mip_Process,\
                         int &Write_Batch, int &Whether_Load_Initial_Point, int &Whether_Record_Process,\
                         char *&Initial_Ig_file_name, char *&Initial_Imd_file_name, char *&Initial_Imp_file_name,\
                         char *&Initial_Beta_mid_file_name, char *&Initial_Beta_mip_file_name)
{
    ifstream read_configure("configure");
    if(!read_configure)
    {
        cout<<"Error: no configuration file is found!"<<endl;
        exit(1);
    }
    string one_line,string_part1,string_part2;
    int i,length;
    while(getline(read_configure,one_line) && one_line.compare("### end"))
    {
        if(!one_line.compare(0,3,"###") || one_line.empty())
            continue;
        length = one_line.length();
        i = one_line.find(" ",0);
        string_part1 = one_line.substr(0,i);
        string_part2 = one_line.substr(i+3,length-i-3);

        // case 1: input of numG
        if(!string_part1.compare("Shared_Dimension"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            numG = atoi(string_part2.c_str());
            if(numG <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }

            continue;
        }
        // case 2: input of numD
        if(!string_part1.compare("Unique_Dimension_In_Matrix1"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            numD = atoi(string_part2.c_str());
            if(numD <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 3: input of numP
        if(!string_part1.compare("Unique_Dimension_In_Matrix2"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            numP = atoi(string_part2.c_str());
            if(numP <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 4: input of input_file_name1
        if(!string_part1.compare("Matrix1_File_Name"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            input_file_name1 = new char[string_part2.length() + 1];
            strcpy(input_file_name1, string_part2.c_str());
            continue;
        }
        // case 5: input of input_file_name2
        if(!string_part1.compare("Matrix2_File_Name"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            input_file_name2 = new char[string_part2.length() + 1];
            strcpy(input_file_name2, string_part2.c_str());
            continue;
        }
        // case 6: input of Co-Module num
        if(!string_part1.compare("Num_Of_Co-Modules"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            M = atoi(string_part2.c_str());
            if(M <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 7: input of Burn In
        if(!string_part1.compare("Burn_In_Iteration"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            Burn_In_Iteration = atoi(string_part2.c_str());
            if(Burn_In_Iteration <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 8: input of Sample Stride
        if(!string_part1.compare("Sample_Stride"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            Sample_Stride = atoi(string_part2.c_str());
            if(Sample_Stride <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 9: input of Total_Num_Of_Sample
        if(!string_part1.compare("Total_Num_Of_Sample"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            Sample_Num = atoi(string_part2.c_str());
            if(Sample_Num <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 10: input of output file1
        if(!string_part1.compare("Indicator_Probability_For_Shared_Dimension"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            output_file_name1 = new char[string_part2.length() + 1];
            strcpy(output_file_name1, string_part2.c_str());
            continue;
        }
        // case 11: input of output file2
        if(!string_part1.compare("Indicator_Probability_For_Unique_Dimension_In_Matrix1"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            output_file_name2 = new char[string_part2.length() + 1];
            strcpy(output_file_name2, string_part2.c_str());
            continue;
        }
        // case 12: input of output file3
        if(!string_part1.compare("Indicator_Probability_For_Unique_Dimension_In_Matrix2"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            output_file_name3 = new char[string_part2.length() + 1];
            strcpy(output_file_name3, string_part2.c_str());
            continue;
        }

        // case 13: input of Whether_Record
        if(!string_part1.compare("Whether_Record_Process"))
        {
            if(string_part2.empty() || !string_part2.compare("-"))
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            Whether_Record_Process = atoi(string_part2.c_str());
            if(Whether_Record_Process != 0 && Whether_Record_Process != 1)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 14: input of Process file for shared dimension
        if(!string_part1.compare("Record_File_For_Indicator_Of_Shared_Dimension"))
        {
            if(string_part2.empty() && Whether_Record_Process == 1 || !string_part2.compare("-") && Whether_Record_Process == 1)
            {
                cout<<"Error: You choose to record process but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            output_file_Ig_Process = new char[string_part2.length() + 1];
            strcpy(output_file_Ig_Process, string_part2.c_str());
            continue;
        }
        // case 15: input of Process file for numD
        if(!string_part1.compare("Record_File_For_Indicator_Of_Unique_Dimension_In_Matrix1"))
        {
            if(string_part2.empty() && Whether_Record_Process == 1 || !string_part2.compare("-") && Whether_Record_Process == 1)
            {
                cout<<"Error: You choose to record process but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            output_file_Imd_Process = new char[string_part2.length() + 1];
            strcpy(output_file_Imd_Process, string_part2.c_str());
            continue;
        }
        // case 16: input of Process file for numP
        if(!string_part1.compare("Record_File_For_Indicator_Of_Unique_Dimension_In_Matrix2"))
        {
            if(string_part2.empty() && Whether_Record_Process == 1 || !string_part2.compare("-") && Whether_Record_Process == 1)
            {
                cout<<"Error: You choose to record process but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            output_file_Imp_Process = new char[string_part2.length() + 1];
            strcpy(output_file_Imp_Process, string_part2.c_str());
            continue;
        }
        // case 17: input of Process file for Beta_mid
        if(!string_part1.compare("Record_File_For_Module_Center_In_Matrix1"))
        {
            if(string_part2.empty() && Whether_Record_Process == 1 || !string_part2.compare("-") && Whether_Record_Process == 1)
            {
                cout<<"Error: You choose to record process but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            output_file_Beta_mid_Process = new char[string_part2.length() + 1];
            strcpy(output_file_Beta_mid_Process, string_part2.c_str());
            continue;
        }
        // case 18: input of Process file for Beta_mip
        if(!string_part1.compare("Record_File_For_Module_Center_In_Matrix2"))
        {
            if(string_part2.empty() && Whether_Record_Process == 1 || !string_part2.compare("-") && Whether_Record_Process == 1)
            {
                cout<<"Error: You choose to record process but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            output_file_Beta_mip_Process = new char[string_part2.length() + 1];
            strcpy(output_file_Beta_mip_Process, string_part2.c_str());
            continue;
        }
        // case 19: input of Process file for Write_Batch
        if(!string_part1.compare("Write_Batch"))
        {
            if(string_part2.empty() && Whether_Record_Process == 1 || !string_part2.compare("-") && Whether_Record_Process == 1)
            {
                cout<<"Error: You choose to record process but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            Write_Batch = atoi(string_part2.c_str());
            if(Write_Batch <= 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }

        // case 20: input for Whether_Load_Initial_Point
        if(!string_part1.compare("Whether_Load_Initial_Point"))
        {
            if(string_part2.empty() || !string_part2.compare("-") )
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            Whether_Load_Initial_Point = atoi(string_part2.c_str());
            if(Whether_Load_Initial_Point != 1 && Whether_Load_Initial_Point != 0)
            {
                cout<<"Error: Please Check the input of "<<string_part1<<endl;
                exit(1);
            }
            continue;
        }
        // case 21: input for Initial_Indicator_For_Shared_Dimension
        if(!string_part1.compare("Initial_Indicator_For_Shared_Dimension"))
        {
            if(string_part2.empty() && Whether_Load_Initial_Point == 1 || !string_part2.compare("-") && Whether_Load_Initial_Point == 1 )
            {
                cout<<"Error: You choose to load initial point but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            Initial_Ig_file_name = new char[string_part2.length() + 1];
            strcpy(Initial_Ig_file_name, string_part2.c_str());
            continue;
        }
        // case 22: input for Initial_Indicator_For_Unique_Dimension_In_Matrix1
        if(!string_part1.compare("Initial_Indicator_For_Unique_Dimension_In_Matrix1"))
        {
            if(string_part2.empty() && Whether_Load_Initial_Point == 1 || !string_part2.compare("-") && Whether_Load_Initial_Point == 1 )
            {
                cout<<"Error: You choose to load initial point but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            Initial_Imd_file_name = new char[string_part2.length() + 1];
            strcpy(Initial_Imd_file_name, string_part2.c_str());
            continue;
        }
        // case 23: input for Initial_Indicator_For_Unique_Dimension_In_Matrix2
        if(!string_part1.compare("Initial_Indicator_For_Unique_Dimension_In_Matrix2"))
        {
            if(string_part2.empty() && Whether_Load_Initial_Point == 1 || !string_part2.compare("-") && Whether_Load_Initial_Point == 1 )
            {
                cout<<"Error: You choose to load initial point but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            Initial_Imp_file_name = new char[string_part2.length() + 1];
            strcpy(Initial_Imp_file_name, string_part2.c_str());
            continue;
        }
        // case 24: input for Initial_Module_Center_In_Matrix1
        if(!string_part1.compare("Initial_Module_Center_In_Matrix1"))
        {
            if(string_part2.empty() && Whether_Load_Initial_Point == 1 || !string_part2.compare("-") && Whether_Load_Initial_Point == 1 )
            {
                cout<<"Error: You choose to load initial point but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            Initial_Beta_mid_file_name = new char[string_part2.length() + 1];
            strcpy(Initial_Beta_mid_file_name, string_part2.c_str());
            continue;
        }
        // case 25: input for Initial_Module_Center_In_Matrix2
        if(!string_part1.compare("Initial_Module_Center_In_Matrix2"))
        {
            if(string_part2.empty() && Whether_Load_Initial_Point == 1 || !string_part2.compare("-") && Whether_Load_Initial_Point == 1 )
            {
                cout<<"Error: You choose to load initial point but not specifying a file: "<<string_part1<<endl;
                exit(1);
            }
            Initial_Beta_mip_file_name = new char[string_part2.length() + 1];
            strcpy(Initial_Beta_mip_file_name, string_part2.c_str());
            continue;
        }

    }

}

void showMemoryInfo(void)
{
    struct rusage usage;
    int who = RUSAGE_SELF;
    int ret;
    ret = getrusage(who, &usage);
    if(ret != -1)
        cout<<usage.ru_idrss<<"\t"<<usage.ru_isrss<<"\t"<<usage.ru_ixrss<<"\t"<<usage.ru_minflt<<"\t"<<usage.ru_nswap<<endl;
    else
        cout<<"error"<<endl;
}

void wait ( double seconds )
{
  clock_t endwait;
  endwait = clock () + seconds * CLOCKS_PER_SEC ;
  while (clock() < endwait) {}
}

void Compute_Posterior_Indicator_Probability(struct_Indicator *Indicator_Array, int length, int M,\
                                             double *Ig_results, double* Imd_results, double* Imp_results)
{
    int mm;

    for(int i = 0; i<(M+1)*numG; i++)
        Ig_results[i] = 0.0;
    for(int i = 0; i<M*numD; i++)
        Imd_results[i] = 0.0;
    for(int i = 0; i< M*numP; i++)
        Imp_results[i] = 0.0;

    for(int i = 0; i<length; i++)
    {
        for(int j = 0; j<numG; j++)
        {
            mm = Indicator_Array[i].Ig[j];
            Ig_results[mm*numG + j] ++;
        }
        for(int j = 0; j < numD; j++)
        {
            for (mm = 1; mm <= M; mm++)
                if(Indicator_Array[i].Imd[j*(M+1) + mm] == 1)
                    Imd_results[(mm-1)*numD + j] ++;
        }
        for(int j = 0; j < numP; j++)
        {
            for (mm = 1; mm <= M; mm++)
                if(Indicator_Array[i].Imp[j*(M+1) + mm] == 1)
                    Imp_results[(mm-1)*numP + j] ++;
        }
    }
    for(int i = 0; i<(M+1)*numG; i++)
        Ig_results[i] = Ig_results[i]/double(length);
    for(int i = 0; i<M*numD; i++)
        Imd_results[i] = Imd_results[i]/double(length);
    for(int i = 0; i< M*numP; i++)
        Imp_results[i] = Imp_results[i]/double(length);
}

//==================================================
int main (int argc, char *argv[])
{

    int Module_Num, Burn_In_Iteration, Sample_Stride, Sample_Num;
    int Whether_Load_Initial_Point,Whether_Record_Process;
    char *input_file_name1, *input_file_name2;
    char *output_file_name1, *output_file_name2, *output_file_name3;
    char *output_file_Ig_Process, *output_file_Imd_Process, *output_file_Imp_Process;
    char *output_file_Beta_mid_Process, *output_file_Beta_mip_Process;
    char *Initial_Ig_file_name, *Initial_Imd_file_name, *Initial_Imp_file_name;
    char *Initial_Beta_mid_file_name, *Initial_Beta_mip_file_name;
    int Write_Batch;

    gsl_set_error_handler_off (); // Turn off the automated GSL error handler.

    read_configure_file(Module_Num,Burn_In_Iteration,Sample_Stride,Sample_Num,\
                        input_file_name1,input_file_name2,\
                        output_file_name1,output_file_name2,output_file_name3,\
                        output_file_Ig_Process, output_file_Imd_Process, output_file_Imp_Process,\
                        output_file_Beta_mid_Process, output_file_Beta_mip_Process,\
                        Write_Batch,Whether_Load_Initial_Point,Whether_Record_Process,\
                        Initial_Ig_file_name,Initial_Imd_file_name,Initial_Imp_file_name,\
                        Initial_Beta_mid_file_name,Initial_Beta_mip_file_name);

    Module_Count.M = Module_Num;
    int M = Module_Num;

    gsl_matrix * Pre_norm_Matrix1 = gsl_matrix_alloc (numD, numG);
    gsl_matrix * Pre_norm_Matrix2 = gsl_matrix_alloc (numP, numG);
    gsl_matrix * Matrix1 = gsl_matrix_alloc (numD, numG);
    gsl_matrix * Matrix2 = gsl_matrix_alloc (numP, numG);

    read_matrix(input_file_name1,Pre_norm_Matrix1);
    read_matrix(input_file_name2,Pre_norm_Matrix2);

    Normalize_Matrix(Pre_norm_Matrix1, Matrix1);
    Normalize_Matrix(Pre_norm_Matrix2, Matrix2);

    gsl_rng * r;
    timeb this_time;
    ftime(&this_time);
    gsl_rng_default_seed = (unsigned long)(this_time.millitm)*getpid();
    r = gsl_rng_alloc(gsl_rng_ranlxd1);

    initialize_hyperparameters(Matrix1,Matrix2);

    struct_Indicator Indicator;
    Indicator.Ig = NULL;
    Indicator.Imd = NULL;
    Indicator.Imp = NULL;
    struct_Module_Center Module_Center;
    Module_Center.Beta_mid = NULL;
    Module_Center.Beta_mip = NULL;

    if(Whether_Load_Initial_Point == 1)
    {
        import_Initial_Interation_Point(Indicator,Module_Center,\
                                        Initial_Ig_file_name,Initial_Imd_file_name,Initial_Imp_file_name,\
                                        Initial_Beta_mid_file_name,Initial_Beta_mip_file_name);
    }
    else
    {
        initialize_Indicator(Indicator,r);
        initialize_Module_Center0(Module_Center);
    }

    SS.SS_mbd = NULL;
    SS.SS_mbp = NULL;
    SS.SS_md = NULL;
    SS.SS_mp = NULL;

    int num_G[M+1];

    compute_SS(Matrix1,Matrix2,Indicator,Module_Center);
    compute_Posterior(Matrix1,Matrix2,Indicator,Module_Center);
    compute_Sum_Num(Indicator);

    // delete the existing files later used as output for record iteration process.
    if(Whether_Record_Process == 1)
    {
        remove(output_file_Ig_Process);
        remove(output_file_Imd_Process);
        remove(output_file_Imp_Process);
        remove(output_file_Beta_mid_Process);
        remove(output_file_Beta_mip_Process);
    }

    //==============================================================
    int Max_Iteration = Burn_In_Iteration + (Sample_Num-1) * Sample_Stride + 1;

    struct_Indicator Indicator_Array[Sample_Num]; // to store the samples after burn in
    struct_Module_Center Module_Center_Array[Sample_Num]; // to store the samples after burn in


    struct_Indicator Indicator_Process[Write_Batch]; // to record the process during iteration
    struct_Module_Center Module_Center_Process[Write_Batch]; // to record the process during iteration

    int process_index = 0;
    for (process_index = 0; process_index<Write_Batch; process_index++)
    {
        Indicator_Process[process_index].Ig = (int*)malloc(sizeof(int)*numG);
        Indicator_Process[process_index].Imd = (int*)malloc(sizeof(int)*numD*(Module_Count.M + 1));
        Indicator_Process[process_index].Imp = (int*)malloc(sizeof(int)*numP*(Module_Count.M + 1));
        Module_Center_Process[process_index].Beta_mid = (double*)malloc(sizeof(double)*numD*(Module_Count.M + 1));
        Module_Center_Process[process_index].Beta_mip = (double*)malloc(sizeof(double)*numP*(Module_Count.M + 1));
    }

    int iter = 0;
    int sample_index = 0;
    while(iter<Max_Iteration)
    {
        process_index = iter%Write_Batch;
        for(int i = 0; i<numG; i++)
            Indicator_Process[process_index].Ig[i] = Indicator.Ig[i];
        for(int i = 0; i<numD*(Module_Count.M+1); i++)
        {
            Indicator_Process[process_index].Imd[i] = Indicator.Imd[i];
            Module_Center_Process[process_index].Beta_mid[i] = Module_Center.Beta_mid[i];
        }
        for(int i = 0; i<numP*(Module_Count.M+1); i++)
        {
            Indicator_Process[process_index].Imp[i] = Indicator.Imp[i];
            Module_Center_Process[process_index].Beta_mip[i] = Module_Center.Beta_mip[i];
        }
        //-------

        cout<<"The "<<iter++<<"th iteration..."<<endl;

        clock_t begin=clock();
        if(iter%Write_Batch == 0 && Whether_Record_Process)
            write_Process_Batch_to_file(Indicator_Process,Module_Center_Process,Write_Batch,M,\
                                        output_file_Ig_Process,output_file_Imd_Process,output_file_Imp_Process,\
                                        output_file_Beta_mid_Process,output_file_Beta_mip_Process);

        //--------------------------------------------------
        for(int j = 0; j<numG; j++)
            Sample_Gene_Indicator(Matrix1,Matrix2,Indicator,Module_Center,j,r);
        compute_num_G(Indicator.Ig,num_G,M);
        //--------------------------------------------------
        for(int mm=1; mm<=Module_Count.M; mm++)
            if(num_G[mm] != 0)
                Sample_Drug_Indicator(Matrix1,Matrix2,Indicator,Module_Center,mm,r);

        //--------------------------------------------------
        for(int mm=1; mm<=Module_Count.M; mm++)
            if(num_G[mm] != 0)
                Sample_Phenotype_Indicator(Matrix1,Matrix2,Indicator,Module_Center,mm,r);
        //----------------------------------------------
        //==================================================
        Change_Drug_Module_Center(Matrix1,Matrix2,Indicator,Module_Center,num_G,r);
        //cout<<"\t"<<iter<<"th "<<"Change Drug Module Center "<<endl;

        Change_Phenotype_Module_Center(Matrix1,Matrix2,Indicator,Module_Center,num_G,r);
        //cout<<"\t"<<iter<<"th "<<"Change Phenotype Module Center "<<endl;

        //================================================
        Exchange_Drug_Indicator(Matrix1,Matrix2,Indicator,Module_Center,num_G,r);
        //cout<<"\t"<<iter<<"th "<<"Propose to Exchange Drug Indicator "<<endl;

        Exchange_Phenotype_Indicator(Matrix1,Matrix2,Indicator,Module_Center,num_G,r);
        //cout<<"\t"<<iter<<"th "<<"Propose to Exchange Phenotype Indicator "<<endl;

        Exchange_Gene_Indicator_Module_NULL(Matrix1,Matrix2,Indicator,Module_Center,r);
        //cout<<"\t"<<iter<<"th "<<"Propose to Exchange Gene Indicator to NULL "<<endl;

        Exchange_Gene_Indicator_Module_Module(Matrix1,Matrix2,Indicator,Module_Center,r);
        //cout<<"\t"<<iter<<"th "<<"Propose to Exchange Gene Indicator M2M "<<endl;

        //===============================================
        Exchange_Module_to_NULL(Matrix1,Matrix2,Indicator,Module_Center,r);
        //cout<<"\t"<<iter<<"th "<<"Propose to Exchange Module to NULL "<<endl;

        compute_SS(Matrix1,Matrix2,Indicator,Module_Center);
        //cout<<"\t"<<iter<<"th "<<"Compute SS "<<endl;
        compute_Posterior(Matrix1,Matrix2,Indicator,Module_Center);
        //cout<<"\t"<<iter<<"th "<<"Compute Posterior "<<endl;
        compute_Sum_Num(Indicator);
        //cout<<"\t"<<iter<<"th "<<"Compute Indicator "<<endl;
        clock_t end=clock();

        cout<<"Time_Elapse: "<<(end-begin)/CLOCKS_PER_SEC<<"s"<<endl;

        if(iter >= Burn_In_Iteration && (iter - Burn_In_Iteration)%Sample_Stride == 0)
        {
            Indicator_Array[sample_index].Ig = (int*)malloc(sizeof(int)*numG);
            Indicator_Array[sample_index].Imd = (int*)malloc(sizeof(int)*numD*(Module_Count.M + 1));
            Indicator_Array[sample_index].Imp = (int*)malloc(sizeof(int)*numP*(Module_Count.M + 1));
            Module_Center_Array[sample_index].Beta_mid = (double*)malloc(sizeof(double)*numD*(Module_Count.M + 1));
            Module_Center_Array[sample_index].Beta_mip = (double*)malloc(sizeof(double)*numP*(Module_Count.M + 1));

            for(int i = 0; i<numG; i++)
                Indicator_Array[sample_index].Ig[i] = Indicator.Ig[i];
            for(int i = 0; i<numD*(Module_Count.M+1); i++)
            {
                Indicator_Array[sample_index].Imd[i] = Indicator.Imd[i];
                Module_Center_Array[sample_index].Beta_mid[i] = Module_Center.Beta_mid[i];
            }
            for(int i = 0; i<numP*(Module_Count.M+1); i++)
            {
                Indicator_Array[sample_index].Imp[i] = Indicator.Imp[i];
                Module_Center_Array[sample_index].Beta_mip[i] = Module_Center.Beta_mip[i];
            }
            sample_index ++;
        }
    }
    if(Whether_Record_Process)
    {
        write_Process_Batch_to_file(Indicator_Process,Module_Center_Process,process_index + 1,M,\
                                output_file_Ig_Process,output_file_Imd_Process,output_file_Imp_Process,\
                                output_file_Beta_mid_Process,output_file_Beta_mip_Process);
    }

    gsl_rng_free(r);
    gsl_matrix_free(Matrix1);
    gsl_matrix_free(Matrix2);

    double *Ig_probability = (double*)malloc(sizeof(double)*(M+1)*numG);
    double *Imd_probability = (double*)malloc(sizeof(double)*M*numD);
    double *Imp_probability = (double*)malloc(sizeof(double)*M*numP);

    Compute_Posterior_Indicator_Probability(Indicator_Array,Sample_Num,Module_Count.M,\
                                            Ig_probability,Imd_probability,Imp_probability);

    write_Ig_probability_to_file(Ig_probability,output_file_name1,Module_Count.M);
    write_Imd_probability_to_file(Imd_probability,output_file_name2,Module_Count.M);
    write_Imp_probability_to_file(Imp_probability,output_file_name3,Module_Count.M);

    free(Ig_probability);
    free(Imd_probability);
    free(Imp_probability);
    for(int i = 0; i< Sample_Num; i++)
    {
        free(Indicator_Array[i].Ig);
        free(Indicator_Array[i].Imd);
        free(Indicator_Array[i].Imp);
        free(Module_Center_Array[i].Beta_mid);
        free(Module_Center_Array[i].Beta_mip);
    }
    return 0;
}
