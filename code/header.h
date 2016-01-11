#ifndef HEADER_H
#define HEADER_H

////////////////////////// Module couts
typedef struct
{
    int M;
} struct_Module_Count;

////////////////////////// SS_parts
typedef struct
{
    double* SS_md;
    double* SS_mbd;
    double* SS_mp;
    double* SS_mbp;
} struct_SS;

/////////////////////////// Posterior_parts
typedef struct
{
    double part1;
    double part2;
    double part3;
} struct_Posterior;
/////////////////////////// Sum_num
typedef struct
{
    int mG;
    int mD;
    int mP;
} struct_Sum_Num;

/////////////////////////// Hyperparameters
typedef struct
{
    double C_D;
    double C_P;
    double C_G;
    double C_M;

    double k_0d;
    double v_0d;
    double S2_0d;
    double k_0p;
    double v_0p;
    double S2_0p;
    double k_taod;
    double v_taod;
    double S2_taod;
    double k_taop;
    double v_taop;
    double S2_taop;

    double k_sigmad;
    double v_sigmad;
    double S2_sigmad;
    double k_sigmap;
    double v_sigmap;
    double S2_sigmap;
} Hyperparameters;

/////////////////////////// Indicators
typedef struct
{
    int *Imd;
    int *Imp;
    int *Ig;
} struct_Indicator;

////////////////////////// Module_Center
typedef struct
{
    double *Beta_mid;
    double *Beta_mip;
} struct_Module_Center;

#endif // HEADER_H
