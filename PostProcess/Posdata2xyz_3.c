

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// THIS VERSION COPIES THE OUTPUT AND INPUT FILE EXACTLY
// NEXT VERSION WILL MAKE MODIFICATIONS
// IT IS NOT OPTIMIZED OR MULTITHREADED, FILES ARE NOT VERY BIG

#include <stdio.h>
#include <string.h>
//#include <iostream>

int main()
{

// read the entire data output file
    int n;
    FILE* dat;
    dat = fopen("Positions.data","r");
    fseek(dat, 0, SEEK_END);
    n = ftell(dat);
    printf("%d\n",n);
    fseek(dat, 0, SEEK_SET);

    char readdata[n];
    fread(readdata,1,n,dat);

    printf("\n");

    for (int i=0;i<n;i++)
    {
        printf("%c",readdata[i]);
    }

    fseek(dat, 0, SEEK_SET);
    fclose(dat);


// read the entire input file
    int m;
    FILE* inp;
    inp = fopen("input","r");
    fseek(inp, 0, SEEK_END);
    m = ftell(dat);
    printf("%d\n",m);
    fseek(inp, 0, SEEK_SET);

    char readinput[m];
    fread(readinput,1,m,inp);

    printf("\n");

    for (int i=0;i<m;i++)
    {
        printf("%c",readinput[i]);
    }

    fseek(inp, 0, SEEK_SET);
    fclose(inp);

// pick out information from input file that is needed to make the modified output file

    int N1;
    sscanf(readinput,"Number_of_atoms_of_element = %d",N1);


// open new and write the modified output file
    FILE* xyz;
    xyz = fopen("Positions.xyz","w");
    fwrite(readdata,1,n,xyz);
    fclose(xyz);

    return 0;
}