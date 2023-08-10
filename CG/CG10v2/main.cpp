

// NOTES
// CG METHOD WORKS
// IMPLIMENTS DIRECT SEARCH


// TODO NEXT

// extend test case to higher dimensionality, n = 100, or up to 100,000
// USE FFTs to calculate gradient of functional, integrals, and anything else can

// minimize a functional that actually means something (physics) quantum mechanics
// translate to GPU

// include bracket, golden section, bisection, and Brent search (use existing Fortran code with interface)


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>

 
//double parab(double* c, double x);
//double invsq3d(double* c, double x, double y, double z);

double rand01();

//void norm(double*** field, int long long n, double N);

//double functional1(double* x, double** Mp, double* B, int long long n);
double functional(double* x, double* dx, double dist, double** Mp, double* B, int long long n);


int main()
{

    int numthreads = 48;

    // check openmp
    printf("number of threads in serial section = %d\n", omp_get_num_threads());
    omp_set_num_threads(numthreads);
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("number of threads in parallel section = %d\n", omp_get_num_threads());
        }
    }


    std::chrono::time_point<std::chrono::system_clock> t1s, t1e, t2s, t2e;
    std::chrono::duration<double> t1se, t2se;


    printf("\n\n");
    printf("---------------------------------------\n");
    printf("START CG test1\n");
    printf("SETUP MATRIX EQUATION WITH KNOWN SOLUTION\n");

    // INITIALIZE
    int long long i1, i2, i3, i4, i5, i6;
    double ptot, pint1, pint2, pext1, pext2;
    double pmin = 1e20;

    // SETUP
    int debug = 0;
    int long long n = 4;   // points in each direction
    double* box = (double*)calloc(2*n,sizeof(double));
    double* distlim = (double*)calloc(2*n,sizeof(double));
    for (i1=0;i1<n;i1++)
    {
        box[i1] = -20.0;
        box[i1+n] = 20.0;
    }

    for (i1=0;i1<n*2;i1++)
    {
        printf("box[%d] = %e\n",i1,box[i1]);
    }

    // X VECTOR
    double* X = (double*)calloc(n,sizeof(double));
    printf("\n");
    for (i1=0;i1<n;i1++)
    {
        X[i1] = (double)(i1+1);
        printf("X[%d] = %e\n",i1,X[i1]);
    }
    printf("\n");


    // M MATRIX
    double* Mvec = (double*)calloc(n,sizeof(double));
    double* M = (double*)calloc(n*n,sizeof(double));
    double** Mp = (double**)calloc(n,sizeof(double*));
    for (i1=0;i1<n;i1++)
    {
        Mp[i1] = &M[i1*n];
        Mvec[i1] = (double)(i1+1);
    }

    for (i1=0;i1<n;i1++)
    {
        for (i2=0;i2<n;i2++)
        {
            Mp[i1][i2] = Mvec[i1]*Mvec[i2];
        }
    }

    printf("M[0][0 to 3] = [%2.2f %2.2f %2.2f %2.2f]\n",Mp[0][0],Mp[0][1],Mp[0][2],Mp[0][3]);
    printf("M[0][0 to 3] = [%2.2f %2.2f %2.2f %2.2f]\n",Mp[1][0],Mp[1][1],Mp[1][2],Mp[1][3]);
    printf("M[0][0 to 3] = [%2.2f %2.2f %2.2f %2.2f]\n",Mp[2][0],Mp[2][1],Mp[2][2],Mp[2][3]);
    printf("M[0][0 to 3] = [%2.2f %2.2f %2.2f %2.2f]\n",Mp[3][0],Mp[3][1],Mp[3][2],Mp[3][3]);
    printf("\n");

    // B VECTOR (CALCULATED BY DEFINITION)
    double* B = (double*)calloc(n,sizeof(double));
    for (i1=0;i1<n;i1++)
    {
        for (i2=0;i2<n;i2++)
        {
            B[i1] += Mp[i1][i2]*X[i2];
        }
        printf("B[%d] = %e\n",i1,B[i1]);
    }

    printf("\n\n");


    //-----------------------------
    // CG METHOD (UNCONSTRAINED) (B = 0 to start)
    printf("---------------------------------------\n");
    printf("RUN CONJUGATE GRADIENT METHOD\n");
    printf("\n");

    //t1s = std::chrono::system_clock::now();

    // SETUP SOLVER
    int it = 1e2;
    int long long itl = 6e7;
    double dx = 1e-6;    // could easily set this to the average, norm/grid points * 1e-5 or something...
    double x[4] = {1.6, 5.4, 3.1, 5.3};   // INITIAL GUESS
    double b[4];  // compare final result
    double sqgradnorm1 = 1.0;   // dont initialize to 0 because will divide by this on first iteration, but wont use it...
    double sqgradnorm2 = 0.0;  
    double sqconjgradnorm = 0.0;
    double Beta = 0.0;
    double maxdneg = 1e10;  // must be careful choosing these starting values
    double maxdpos = 1e10;
    int long long resetsteps = n;
    int long long steps = resetsteps-1;  // set to n*2-1 on 1st iteration so will run STEEPEST DESCENT on first

    double val, val1, val2, valmin;
    double xd1[4];
    double xd2[4];
    double grad[4];
    double conjgrad[4];
    double conjgradprev[4];
    double ones[4] = {1, 1, 1, 1};
    double one[4] = {0, 0, 0, 0};
    double dist = 0;


    int* parafoundmin = (int*)calloc(numthreads,sizeof(int));  // need to change to malloc and move before the for loop iterations
    int long long* paraitmin = (int long long*)calloc(numthreads,sizeof(int long long));
    double* paramin = (double*)calloc(numthreads,sizeof(double));  // HAD TO TAKE OUT STATIC
    

    // reduced global min holders
    int long long itmin = 0;
    int foundmin = 0;
    double disttest = 0.0;


    // CALCULATE
    // calculate value of functional at initial guess point
    val = functional(x,ones,0.0,Mp,B,n);
    // compare value to minimum tolerance
    if (val<valmin)
    {
        valmin = val;
    }

    printf("INITIAL GUESS: x=[%2.3f %2.3f %2.3f %2.3f] (val = %2.3e)\n",x[0],x[1],x[2],x[3],val);
    printf("\n");

    t1s = std::chrono::system_clock::now();

    for (i1=0;i1<it;i1++)
    {
        sqgradnorm2 = 0;
        sqconjgradnorm = 0;

        // CALCULATE GRADIENT VECTOR AND NORMALIZE TO GET UNIT VECTOR IN GRADIENT DIRECTION
        for (i2=0;i2<n;i2++)
        {
            one[i2] += 1;
            val1 = functional(x,one,-dx,Mp,B,n);  // evaluate functional at offsets for partial derivatives
            val2 = functional(x,one,dx,Mp,B,n);  // evaluate functional at offsets for partial derivatives
            one[i2] -= 1;
            grad[i2] = (val2-val1)/(2.0*dx);  // partial derivative for each variable separately at the point
            //sqgradnorm2 += pow(grad[i2],2.0);
            sqgradnorm2 += grad[i2]*grad[i2];
        }
        for (i2=0;i2<n;i2++)
        {
            grad[i2] /= sqrt(sqgradnorm2);  // normalize the gradient
        }

        Beta = sqgradnorm2/sqgradnorm1; // find new conjugate direction coefficient (Fletcher-Reeves)
        sqgradnorm1 = sqgradnorm2;

        steps++;
        if (steps == resetsteps)     // STEEPEST DESCENT DIRECTION reset conjugate gradient direction to steepest
        {
            for (i2=0;i2<n;i2++)
            {
                conjgrad[i2] = -grad[i2];
                conjgradprev[i2] = conjgrad[i2];
            }
            steps = 0;
        }
        else
        {
            for (i2=0;i2<n;i2++)
            {
                conjgrad[i2] = -grad[i2] + Beta*conjgradprev[i2];   // CONJUGATE GRADIENT METHOD IMPLIMENTATION
                sqconjgradnorm += conjgrad[i2]*conjgrad[i2];
            }
            for (i2=0;i2<n;i2++)
            {
                conjgrad[i2] /= sqrt(sqconjgradnorm);         // normalize the conjugate gradient
                conjgradprev[i2] = conjgrad[i2];
            }
        }

        // PREP FOR LINE MINIMIZATION OPTIMIZATION ALONG VECTOR DIRECTION
        // calc maximum distance before one of domain variables goes negative (all values should be +)
        // calc maximum distance in + and - directions before one of domain variables goes past box domain boundary

        for (i2=0;i2<n;i2++)
        {
            distlim[i2] = abs((box[i2]-x[i2])/conjgrad[i2]);
            distlim[i2+n] = abs((box[i2+n]-x[i2])/conjgrad[i2]);
        }
        // + Distance Direction
        maxdpos = 1e10;
        maxdneg = 1e10;
        for (i2=0;i2<n;i2++)
        {
            if (conjgrad[i2] > 0)
                if (distlim[i2+n] < maxdpos)   // + dist, + conj gradient, use + high boundary
                    maxdpos = distlim[i2+n];
            if (conjgrad[i2] < 0)
                if (distlim[i2] < maxdpos)     // + dist, - conj gradient, use - low boundary
                    maxdpos = distlim[i2];
        }
        // - Distance Direction
        for (i2=0;i2<n;i2++)
        {
            if (conjgrad[i2] > 0)
                if (distlim[i2] < maxdneg)   // - dist, + conj gradient, use - low boundary
                    maxdneg = distlim[i2];
            if (conjgrad[i2] < 0)
                if (distlim[i2+n] < maxdneg)     // - dist, - conj gradient, use + high boundary
                    maxdneg = distlim[i2+n];
        }


        //--------------------------------------------------------
        // MULTIDIMENSIONAL VECTOR DIRECTION LINE MINIMIZATION
        // this will be a separate function

        for (i2=0;i2<numthreads;i2++)
        {
            parafoundmin[i2] = 0;
            paraitmin[i2] = 0;  // setting this to 0 is good because it just keeps the starting point
            paramin[i2] = valmin;
        }

        // reduced global min holders
        int long long itmin = 0;
        int foundmin = 0;
        double disttest = 0.0;

        #pragma omp parallel
        {
            int long long ii;
            int mynum = omp_get_thread_num();
            double myval;
            double myvalmin = valmin;
            double myfoundmin = 0;
            int long long myitmin = 0;

            int long long is = mynum*itl/numthreads;
            int long long ie = is+itl/numthreads;

            for (ii=is;ii<ie;ii++)
            {
                myval = functional(x,conjgrad,(maxdpos+maxdneg)/((double)(itl-1))*ii-maxdneg,Mp,B,n);
                if (myval<myvalmin)
                {
                    myfoundmin = 1;
                    myvalmin = myval;
                    myitmin = ii;
                }
                
                if (ii==ie-1)
                {
                    parafoundmin[mynum] = myfoundmin;
                    paramin[mynum] = myvalmin;
                    paraitmin[mynum] = myitmin;
                }
            }
        }

        // REDUCE MINIMUM OPERATION (needed to keep itmin associated with val so did it myself)
        for (i2=0;i2<numthreads;i2++)
        {
            if (parafoundmin[i2] == 1)
            {
                if (paramin[i2] < valmin)
                {
                    foundmin = 1;
                    valmin = paramin[i2];
                    itmin = paraitmin[i2];
                }
            }
        }

        if (foundmin == 1)
        {
            disttest = (maxdpos+maxdneg)/((double)(itl-1))*itmin-maxdneg;
            for (i3=0;i3<n;i3++)
            {
                x[i3] += conjgrad[i3]*disttest;    // set x equal to the minimum found x
            }
        }

        printf("%d: x(dist = %2.4e) = [%2.10f %2.10f %2.10f %2.10f] (val = %2.3e)\n",i1, disttest,x[0],x[1],x[2],x[3],valmin);
    }

    t1e = std::chrono::system_clock::now();
    t1se = t1e-t1s;

    printf("\n");
    printf("MINIMUM FOUND: (%e)\n",abs(valmin));
    printf("DURATION (seconds) = %e\n\n",t1se);

    // calc b
    for (i1=0;i1<n;i1++)
    {
        b[i1] = 0;
        for (i2=0;i2<n;i2++)
        {
            b[i1] += Mp[i1][i2]*x[i2];
        }
    }

    printf("B calc = %e %e %e %e\n",b[0],b[1],b[2],b[3]);
    printf("B orig = %e %e %e %e\n",B[0],B[1],B[2],B[3]);
    printf("error %1.3e %1.3e %1.3e %1.3e\n",(b[0]-B[0])/B[0],(b[1]-B[1])/B[1],(b[2]-B[2])/B[2],(b[3]-B[3])/B[3]);
    printf("\n");

    printf("X calc = %e %e %e %e\n",x[0],x[1],x[2],x[3]);
    printf("X orig = %e %e %e %e\n",X[0],X[1],X[2],X[3]);
    printf("error %1.3e %1.3e %1.3e %1.3e\n",(x[0]-X[0])/X[0],(x[1]-X[1])/X[1],(x[2]-X[2])/X[2],(x[3]-X[3])/X[3]);
    printf("\n");


    return 42;
}


double functional(double* x, double* dx, double dist, double** Mp, double* B, int long long n)
{
    // this functional takes an input (x) offset direction and distance along that direction
    // 1/2 * x^T * M * x - x^T * b
    // b = 0

    double out = 0;

    for (int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            out += 0.5*((x[j]+dx[j]*dist)*Mp[i][j]*(x[i]+dx[i]*dist));
        }
        out -= (x[i]+dx[i]*dist)*B[i];
    }

    return out;
}


double rand01()
{
    return (double)rand()/((double)RAND_MAX) * (double)rand()/((double)RAND_MAX) * (double)rand()/((double)RAND_MAX);
}