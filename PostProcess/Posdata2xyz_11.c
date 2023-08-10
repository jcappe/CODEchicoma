

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// ALL DATA POINTS WRITTEN
// NEED TO ADD ATOMIC IDENTIFIER LETTERS
// TWO DIFFERENT METHODS POSSIBLE
// INSERT LETTERS, MUCH BETTER, NO PARSING OF DATA BLOCKS (EACH TIME STEP WHOLE WRITE)
// OR COULD ADD LETTER THEN ADD EACH LINE INDIVIDUALLY, NOT GOOD WILL TRY TO AVOID
// NEVER MIND THERE IS NO INSERT!

#include <stdio.h>
#include <string.h>

int getintpow(int x);


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

    char* elist = strstr(readinput,"&element_list");

    // READ NUMBER OF ATOMS OF EACH ELEMENT
    char* nat = strstr(&elist[1],"Number_of_atoms_of_element");
    char* nat2 = strstr(nat,"=");
    char* endline = strstr(nat2,"\n");
    char* newspace = strstr(nat2," ");
    char* space;
    int Ntemp;
    int Ntot = 0;
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
        Ntot += Ntemp;
    }

    // READ ATOM IDENTIFIER LABELS FROM PP_file &element_list
    // RIGHT NOW ONLY DOES SINGLE LETTER IDENTIFIERS
    char* PPfile = strstr(elist,"PP_file");
    char* PPfile2 = strstr(PPfile,"=");
    char* endline2 = strstr(PPfile2,"\n");
    char* newspace2 = strstr(PPfile2," ");
    char* space2;
    char AIDtemp;
    int AIDn = 0;
    
    while(newspace2 < endline2)
    {
        AIDn += 1;
        sscanf(&newspace2[1],"%*c%*c%c",&AIDtemp);
        space2 = newspace2;
        newspace2 = strstr(&space2[1]," ");
    }

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
    }

    // PRINT OUT RESULTS
    printf("\n");
    printf("Total Number of Atoms = %d\n",Ntot);
    printf("Number of Types of Atoms = %d\n",Nn);
    for (int ai = 0; ai<Nn; ai++)
    {
        printf("Number of Atoms of Type (%c) = %d\n",AID[ai],N[ai]);
    }
    printf("\n");


// open new and write the modified output file
    FILE* xyz;
    xyz = fopen("Positions.xyz","w");

    // get the number of time steps
    int nt;  
    char* tdep = strstr(readinput,"&Time_Dependent");
    char* ntc = strstr(tdep,"nt");
    char* ntc2 = strstr(&ntc[2],"=");
    sscanf(&ntc2[1],"%d", &nt);
    printf("num time steps = %d\n\n",nt);

    char* Posdat1;
    char* Posdat2;
    int t1, t2;

    Posdat1 = strstr(readdata,"Time (a.u):");

    char Ntotc[getintpow(Ntot)];
    sprintf(Ntotc,"%d",Ntot);

    //nt = 3;
    for (int tt=0;tt<nt;tt++)
    {
        fwrite(Ntotc,1,getintpow(Ntot),xyz);
        fwrite("\n",1,1,xyz);

        Posdat2 = strstr(&Posdat1[1],"Time (a.u):");
        sscanf(Posdat1,"%*s%*s%d", &t1);
        sscanf(Posdat2,"%*s%*s%d", &t2);
        fwrite(Posdat1,1,(Posdat2-Posdat1-3),xyz);
        if (tt == nt-1)
        {
            fwrite(Ntotc,1,getintpow(Ntot),xyz);
            fwrite("\n",1,1,xyz);
            fwrite(Posdat2,1,(Posdat2-Posdat1-3),xyz);
        }

        printf("tt = %d\n",tt);
        printf("t1 = %d\n",t1);
        printf("t2 = %d\n",t2);
        printf("diff = %d\n",Posdat2 - Posdat1);
        printf("\n");

        Posdat1 = Posdat2;
    }

    fclose(xyz);

    return 0;
}



int getintpow(int x)
{
    if (0<x && x<10)
        return 1;
    if (9<x && x<100)
        return 2;
    if (99<x && x<1000)
        return 3;
    if (999<x && x<10000)
        return 4;
    if (9999<x && x<100000)
        return 5;
    if (99999<x && x<1e6)
        return 6;
    if (999999<x && x<1e7)
        return 7;
    if (9999999<x && x<1e8)
        return 8;
    if (99999999<x && x<1e9)
        return 9;
}