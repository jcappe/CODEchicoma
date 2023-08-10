

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// IT WORKS!
// CAN READ ANY NUMBER OF TYPES OF ATOMS
// ANY NUMBER OF EACH ONE IN ORDER
// READS IN SINGLE CHARACTER ATOM IDENTIFIERS FROM PP_file list

#include <stdio.h>
#include <string.h>

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

    /*
    printf("\n");
    printf("\n");
    for (int i=0;i<n;i++)
    {
        printf("%d ",readinput[i]);
    }
    printf("\n");
    printf("\n");
    */

    fclose(inp);


// pick out information from input file that is needed to make the modified output file

    char* elist = strstr(readinput,"&element_list");
    //printf("%s\n",elist);

    // READ NUMBER OF ATOMS OF EACH ELEMENT
    char* nat = strstr(&elist[1],"Number_of_atoms_of_element");
        //printf("%s\n",nat);
    char* nat2 = strstr(nat,"=");
        //printf("%s\n",nat2);
    char* endline = strstr(nat2,"\n");
        //printf("%s\n",endline);
    char* newspace = strstr(nat2," ");
        //printf("%s\n",&newspace[1]);
    char* space;
    int Ntemp;
    int Nn = 0; // initialize the number of types of atoms to be used as counter
    while(newspace < endline)
    {
        Nn += 1;
        sscanf(&newspace[1],"%d",&Ntemp);
        space = newspace;
        newspace = strstr(&space[1]," ");
    }
    int N[Nn];
    int Nni = -1; // number of types of atoms index
    newspace = strstr(nat2," "); // reset the pointer
    while(newspace < endline)
    {
        Nni += 1;
        sscanf(&newspace[1],"%d",&Ntemp);
        space = newspace;
        newspace = strstr(&space[1]," ");
        N[Nni] = Ntemp;
        //printf("%d\n",Ntemp);
    }

    for (int ai = 0; ai<Nn; ai++)
    {
        //printf("%d\n",N[ai]);
    }
    //printf("\n");

    // READ ATOM IDENTIFIER LABELS FROM PP_file &element_list
    // RIGHT NOW ONLY DOES SINGLE LETTER IDENTIFIERS
    char* PPfile = strstr(elist,"PP_file");
    //printf("%s\n",PPfile);
    char* PPfile2 = strstr(PPfile,"=");
    //printf("%s\n",PPfile2);
    char* endline2 = strstr(PPfile2,"\n");
    char* newspace2 = strstr(PPfile2," ");
    //printf("diff = %d\n",endline2 - newspace2);
    char* space2;
    char AIDtemp;
    int AIDn = 0; // 
    
    while(newspace2 < endline2)
    {
        AIDn += 1;
        sscanf(&newspace2[1],"%*c%*c%c",&AIDtemp);
        space2 = newspace2;
        newspace2 = strstr(&space2[1]," ");
        //printf("AIDtemp = %c\n",AIDtemp);
    }
    //printf("\n");

    char AID[AIDn];
    int AIDni = -1; // number of types of atoms index
    newspace2 = strstr(PPfile2," "); // reset the pointer
    while(newspace2 < endline2)
    {
        AIDni += 1;
        sscanf(&newspace2[1],"%*c%*c%c",&AIDtemp);
        space2 = newspace2;
        newspace2 = strstr(&space2[1]," ");
        AID[AIDni] = AIDtemp;
        //printf("AID = %c\n",AID[AIDni]);
    }

    // PRINT OUT RESULTS
    printf("Number of Types of Atoms = %d\n",Nn);
    for (int ai = 0; ai<Nn; ai++)
    {
        //printf("Number of Atoms of Type (%s) = %d\n",AID[ai],N[ai]);
        printf("Number of Atoms of Type (%c) = %d\n",AID[ai],N[ai]);
    }


// open new and write the modified output file
    FILE* xyz;
    xyz = fopen("Positions.xyz","w");
    fwrite(readdata,1,n,xyz);
    fclose(xyz);

    return 0;
}