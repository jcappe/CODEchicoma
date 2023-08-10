

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// GOT THE TWO NUMBERS, should work for any number of elements

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
    fseek(dat, 0, SEEK_SET);
    char readdata[n];
    fread(readdata,1,n,dat);
    fclose(dat);


// read the entire input file
    int m;
    FILE* inp;
    inp = fopen("input","r");
    fseek(inp, 0, SEEK_END);
    m = ftell(dat);
    fseek(inp, 0, SEEK_SET);
    char readinput[m];
    fread(readinput,1,m,inp);

    fclose(inp);


// pick out information from input file that is needed to make the modified output file

    char* nat = strstr(readinput,"Number_of_atoms_of_element");
    char* nat2 = strstr(nat,"=");
    char* endline = strstr(nat2,"\n");
    //printf("%d\n",endline-nat2);
    char* space;
    char* newspace = strstr(nat2," ");
    while(newspace < endline)
    {
        printf("number: %s\n",&newspace[1]);
        space = newspace;
        newspace = strstr(&space[1]," ");
    }

    //int N;
    //sscanf(nat2,"%*s%d",&N);
    //printf("N = %d\n",N);


// open new and write the modified output file
    FILE* xyz;
    xyz = fopen("Positions.xyz","w");
    fwrite(readdata,1,n,xyz);
    fclose(xyz);

    return 0;
}