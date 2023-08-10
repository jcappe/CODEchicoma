

// THIS C CODE READS
// Positions.data
// AND WRITES
// Positions.xyz
// FOR OPENING IN PARAVIEW AS A MOVIE FILE
// WITH THE "XYZ Reader" Reader

#include <stdio.h>
#include <string.h>
//#include <iostream>

int main()
{

    printf("\n");
    printf("hello!\n");
    printf("\n");

    const int n = 1000;

    FILE* dat;
    dat = fopen("Positions.data","r");
    char readdata[n];

    //fseek(dat, 0, SEEK_SET);
    fread(readdata,1,n,dat);

    for (int i=0;i<n;i++)
    {
        printf("%d ",readdata[i]);
    }

    printf("\n");
    printf("\n");

    for (int i=0;i<n;i++)
    {
        printf("%c",readdata[i]);
    }

    printf("\n");
    printf("\n");

    return 0;
}