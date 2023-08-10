

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

// THIS ONE COPIES THE FILE EXACTLY
// NEXT VERSION WILL MAKE MODIFICATIONS
// IT IS NOT OPTIMIZED OR MULTITHREADED, FILES ARE NOT VERY BIG

#include <stdio.h>
#include <string.h>
//#include <iostream>

int main()
{
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

    FILE* xyz;
    xyz = fopen("Positions.xyz","w");
    fwrite(readdata,1,n,xyz);
    fclose(xyz)

    return 0;
}