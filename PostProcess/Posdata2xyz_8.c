

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// IT WORKS!
// STORING THE NUMBER OF ATOMS AND THE NUMBER OF TYPES IN ARRAY
// NOW CAN USE TO PRINT MODIFIED OUTPUT FILE
// JUST NEED PP_file ATOM IDENTIFIER LETTERS TO BE READ PROPERLY...


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

    // READ NUMBER OF ATOMS OF EACH ELEMENT
    char* nat = strstr(readinput,"Number_of_atoms_of_element");
    char* nat2 = strstr(nat,"=");
    char* endline = strstr(nat2,"\n");
    char* newspace = strstr(nat2," ");
    char* space;
    int Ntemp;
    int Nn = 0; // initialize the number of types of atoms to be used as counter
    while(newspace < endline)
    {
        Nn += 1;
        sscanf(&newspace[1],"%d",&Ntemp);
        space = newspace;
        newspace = strstr(&space[1]," ");
        printf("Natoms temp = %d\n",Ntemp);
    }
    printf("Number of Types of Atoms = %d\n",Nn);
    int N[Nn];
    int Nni = -1; // number of types of atoms index
    newspace = strstr(nat2," "); // reset the pointer
    while(newspace < endline)
    {
        Nni += 1;
        sscanf(&newspace[1],"%d",&Ntemp);
        space = newspace;
        newspace = strstr(&space[1]," ");
        printf("Natoms temp = %d\n",Ntemp);
        N[Nni] = Ntemp;
        printf("Natoms temp = %d\n",N[Nni]);
    }
    for (int ai = 0; ai<Nn; ai++)
    {
        printf("Number of Atoms of Type (%s) = %d\n","TYPE",N[ai]);
    }


// open new and write the modified output file
    FILE* xyz;
    xyz = fopen("Positions.xyz","w");
    fwrite(readdata,1,n,xyz);
    fclose(xyz);

    return 0;
}