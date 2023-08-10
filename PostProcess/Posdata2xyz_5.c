

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// FIGURED OUT HOW TO SEARCH AND FIND A SUBSTRING INSIDE A STRING
// GIVES THE POINTER IN MEMORY TO THE START OF THE MATCHED SEQUENCE OF CHARACTERS
// CALLING PRINTF %S FROM THAT POINTER WILL PRINT ALL FOLLOWING CHARACTERS INCLUDING NEW LINES

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

    printf("\n");
    for (int i=0;i<m;i++)
    {
        printf("%d ",readinput[i]);
    }
    printf("\n\n");

    fclose(inp);


// pick out information from input file that is needed to make the modified output file

    // test string
    char* test = "number of atoms 4";
    char Ntest;
    char testout[30];

    sscanf(test,"number of atoms %d",&Ntest);
    printf("%d\n",Ntest);

    sscanf(test,"number of atoms %c",testout);
    printf("%c\n",testout[0]);

    printf("\n");

///////////////



    char* nat = strstr(readinput,"Number_of_atoms");
    printf("found substring inside string: %s\n",nat);


    char N1;
    sscanf(readinput,"Number_of_atoms_of_element = %d",&N1);
    printf("ATOMS OF TYPE 1: %d\n",N1);


// open new and write the modified output file
    FILE* xyz;
    xyz = fopen("Positions.xyz","w");
    fwrite(readdata,1,n,xyz);
    fclose(xyz);

    return 0;
}