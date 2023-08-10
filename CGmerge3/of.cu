

////////////////////////////////////////
// INCLUDES
////////////////////////////////////////

#include <numeric>
#include <vector>
#include <complex>
#include <cufftMp.h>
#include <mpi.h>
#include <math.h>
#include <thrust/complex.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cuComplex.h>

#include "/users/pjlohr/cuda_examples/CUDALibrarySamples/cuFFTMp/samples/common/error_checks.hpp"
#include "/users/pjlohr/cuda_examples/CUDALibrarySamples/cuFFTMp/samples/iterators/box_iterator.hpp"


////////////////////////////////////////
// DEFINES NAMESPACES
////////////////////////////////////////

#define COMPLEX cufftDoubleComplex
#define REAL double
#define CUDA_CHECK_ERROR(val) check((val), #val, __FILE__, __LINE__)

using namespace std;


////////////////////////////////////////
// STRUCTURES TEMPLATES
////////////////////////////////////////

// FORTRAN TO C++ for loop index trans:
// do i=1,nx ... (i,j,k) -> for (int i=0;i<nx;i++) ... [k+nx*j+nx*ny*i]
// extern "C" {
//    void Orbital_free(int, int, int, REAL *, COMPLEX *, COMPLEX*, MPI_Fint *);
// }
// void Orbital_free(int argc, char **argv, int nx, int ny, int nz, REAL *cutwf, COMPLEX *_psiR_, COMPLEX *_psiG_, MPI_Fint *F_COMM,

struct energies_struct {
    REAL hartree;
    REAL ion_local;
    REAL kinetic_local;
    REAL kinetic;
    REAL total;
    REAL xc;
};
struct potentials_struct {
    REAL hartree;
    REAL ion_local;
    REAL kinetic_local;
    REAL kinetic;
    REAL total;
    REAL xc;
    REAL external;
};

template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    }
}


////////////////////////////////////////
// FUNCTION DECLARES
////////////////////////////////////////

// SETUP INITIALIZE FINALIZE
int initialize(int argc, char **argv);
unsigned int setup_fft(size_t nx, size_t ny, size_t nz, MPI_Comm comm);
void finalize(extpot, rho, psi, descx, descy, descz, unsigned int plan);

// MAIN FUNCTIONS (TYPICALLY CALLED BY EXTERNAL PROGRAM)
REAL energy calc_value();    // NEED TO FIGURE OUT WHAT TO GIVE AS INPUT PARAMETERS!!
//REAL* calc_derivative = ();;

// POTENTIALS
void calc_thomas_fermi(size_t my_data_size, COMPLEX *d_rho_R, REAL *d_tf_pot, REAL *d_el_local_kinetic_energy,
                        int rank, int size, MPI_Comm comm, int ndevices, REAL dx, REAL dy, REAL dz, REAL system_temperature, int density_ns,  REAL tf_gamma);
void calc_vw(size_t my_data_size, size_t nx, size_t ny, size_t nz, cudaLibXtDesc *psi_G_desc, REAL *d_vw_pot, REAL *d_el_kinetic_energy,
                cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, int rank, int size, MPI_Comm comm, unsigned int plan,
                 REAL Lx, REAL Ly, REAL Lz, REAL cutoff, REAL tf_lambda);
void calc_local_pp_energy(size_t my_data_size, COMPLEX d_rho_R, REAL *local_kinetic_pot,
                            int rank, int size, MPI_Comm comm, int ndevices);
void calc_hartree(size_t my_data_size, size_t nx, size_t ny, size_t nz, cudaLibXtDesc *rho_R_desc, cudaLibXtDesc *rho_G_desc, REAL *d_hartree_pot, REAL *d_el_hartree_energy,
                    cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, int rank, int size, MPI_Comm comm, unsigned int plan, REAL dx, REAL dy, REAL dz);
void calc_xc_vwn(size_t my_data_size, COMPLEX *d_rho_R, REAL *d_xc_pot, REAL *d_el_xc_energy, int rank, REAL dx, REAL dy, REAL dz, REAL c_light);

// POTENTLAL KERNELS
__global__ void tf_pot_kernel(COMPLEX *d_rho_R, REAL *d_local_kinetic_pot, REAL *d_elemental_local_kinetic_energy, int data_size, REAL system_temperature, 
                            REAL dx, REAL dy, REAL dz, int density_ns, REAL tf_gamma);
__global__ void vw_kernel(BoxIterator<COMPLEX> d_psi_G_begin, BoxIterator<COMPLEX> d_psi_G_end, int rank, int size, size_t nx, size_t ny, size_t nz,
                            REAL *kx, REAL *ky, REAL *kz, REAL *d_kinetic_energy, REAL Lx, REAL Ly, REAL Lz, REAL cutoff, REAL tf_lambda);
void calc_local_pp_energy_kernels(size_t my_data_size, COMPLEX d_rho_R, REAL *local_kinetic_pot,
                            int rank, int size, MPI_Comm comm, int ndevices);
__global__ void hartree_kernel(BoxIterator<COMPLEX> d_rho_G_begin, BoxIterator<COMPLEX> d_rho_G_end, int rank, int size, REAL *kx, REAL *ky, REAL *kz);
__global__ void hartree_energy_kernel(COMPLEX *d_rho_R, COMPLEX *d_hartree_pot_desc, REAL *d_hartree_pot, REAL *d_elemental_hartree_energy, int data_size, REAL dx, REAL dy, REAL dz);
__global__ void calc_xc_vwn_kernel(COMPLEX *d_rho_R, REAL *d_vxc, REAL *d_exc, REAL *d_elemental_xc_energy, size_t my_data_size,  
                REAL dx, REAL dy, REAL dz, REAL y0, REAL b, REAL c, REAL A, bool relat, REAL c_light);

// OTHER KERNELS (WAVEFUNCTION DENSITY NORMS)
__global__ void calc_psi(COMPLEX *d_rho_R, COMPLEX *d_psi_R, REAL *d_psi_R_sq_dr, size_t my_data_size, REAL dx, REAL dy, REAL dz);
__global__ void calc_density(COMPLEX *d_psi_R, COMPLEX *d_rho_R, size_t my_data_size);
__global__ void calc_norm1(COMPLEX *d_psi_R, REAL *d_psi_R_sq_dr, size_t my_data_size, REAL dx, REAL dy, REAL dz);
__global__ void calc_norm2(COMPLEX *d_psi_R, COMPLEX *d_psi_G, REAL *d_psi_R_sq_dr, REAL psi_norm_before, size_t my_data_size, size_t nelec, REAL dx, REAL dy, REAL dz);
void calc_norm_and_psi(COMPLEX *rho_R, COMPLEX *psi_R, cudaLibXtDesc *rho_R_desc, cudaLibXtDesc *rho_G_desc, cudaLibXtDesc *psi_R_desc, cudaLibXtDesc *psi_G_desc, 
                    size_t my_data_size, size_t nelec, REAL dx, REAL dy, REAL dz, unsigned int plan, int rank, int size, size_t nx, size_t ny, size_t nz, 
                    cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, REAL cutoff, MPI_Comm comm);
__global__ void scaling_kernel(BoxIterator<COMPLEX> d_begin, BoxIterator<COMPLEX> d_end, int rank, int size, size_t nx, size_t ny, size_t nz)

// SUPPORTING FUNCTIONS FOR KERNELS
__device__ REAL elem(REAL y, bool deriv = false, bool sec_deriv = false);

// MATH FUNCTIONS
int displacement(int length, int rank, int size);
__device__ REAL get_Y(REAL y, REAL b, REAL c);

// GRID
void gen_ks (size_t nx, size_t ny, size_t nz, REAL Lx, REAL Ly, REAL Lz, REAL *kx, REAL *ky, REAL *kz,
            cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, unsigned int cufft_plan);

// PRINT FUNCTIONS
void printdoubleArray(REAL* arr, int size);


////////////////////////////////////////
// INITIALIZATION AND FINALIZATION (EXTERNAL CONTROLING MAIN CODE)
////////////////////////////////////////

int initialize(int argc, char **argv)
{

    // SETUP
    COMPLEX *rho_R, COMPLEX *psi_R, 
    cudaLibXtDesc *rho_R_desc, cudaLibXtDesc *rho_G_desc, cudaLibXtDesc *psi_R_desc, cudaLibXtDesc *psi_G_desc, 
    size_t my_data_size, size_t nelec, REAL dx, REAL dy, REAL dz, 
    unsigned int plan, int rank, int size, size_t nx, size_t ny, size_t nz, 
    cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, REAL cutoff, MPI_Comm comm

    // Placeholder input variables
    bool calc_derivative = true;
    bool calc_value = true;
    bool vW = true;

    REAL cutoff = 10.0;
    size_t nelec = 10;
    REAL system_temperature = 100;
    int density_ns = 1;

    REAL c_light = 1.0;
    REAL tf_gamma = 0.75;
    REAL tf_lambda = 0.25;

    REAL Lx = 10;
    REAL Ly = 10;
    REAL Lz = 10;

    REAL dg_x = 2.0*M_PI/Lx;
    REAL dg_y = 2.0*M_PI/Ly;
    REAL dg_z = 2.0*M_PI/Lz;

    // setup MPI comm
    MPI_Init(4,4);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // find cuda devices
    int ndevices;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));

    // printf("%d cuda devices\n", ndevices);
    size_t nx = (argc >= 2 ? atoi(argv[1]) : 8 * size); // any value >= size is OK
    size_t ny = (argc >= 2 ? atoi(argv[1]) : 8 * size); // any value >= size is OK
    size_t nz = (argc >= 2 ? atoi(argv[1]) : 8 * size); // any value >= size is OK

    REAL dx = Lx / nx;
    REAL dy = Ly / ny;
    REAL dz = Lz / nz;

    /**
     * This samples illustrates a basic use of cuFFTMp using the built-in, optimized, data distributions.
     * 
     * It assumes the CPU data is initially distributed according to CUFFT_XT_FORMAT_INPLACE, a.k.a. X-Slabs.
     * Given a global array of size X * Y * Z, every MPI rank owns approximately (X / ngpus) * Y * Z entries.
     * More precisely, 
     * - The first (X % ngpus) MPI rank each own (X / ngpus + 1) planes of size Y * Z,
     * - The remaining MPI rank each own (X / ngpus) planes of size Y * Z
    */
    int ranks_cutoff = nx % size;
    size_t my_nx = (nx / size) + (rank < ranks_cutoff ? 1 : 0);
    size_t my_ny = ny;
    size_t my_nz = nz;
    size_t my_data_size = my_nx * my_ny * my_nz;
    
    printf("Hello from rank %d/%d using GPU %d transform of size %zu x %zu x %zu, local size %zu x %zu x %zu\n", rank, size, rank % ndevices, nx, ny, nz, my_nx, my_ny, my_nz);

    // Generate local, distributed, density data based on X slabs
    COMPLEX *rho_R, *psi_R;
    rho_R = (COMPLEX*)malloc(my_data_size * sizeof(COMPLEX)); 
    psi_R = (COMPLEX*)malloc(my_data_size * sizeof(COMPLEX)); 

    REAL *d_hartree_pot, *d_xc_pot, *d_ext_pot, *d_tf_pot, *d_vw_pot, *d_localPP_pot;
    CUDA_CHECK_ERROR(cudaMalloc(&d_hartree_pot, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_xc_pot, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_ext_pot, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_tf_pot, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_vw_pot, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_localPP_pot, my_data_size * sizeof(REAL)));

    // Malloc "elemental" energy vectors on GPU
    REAL *d_el_hartree_energy, *d_el_xc_energy, *d_el_tf_energy, *d_el_kinetic_energy;
    CUDA_CHECK_ERROR(cudaMalloc(&d_el_hartree_energy, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_el_xc_energy, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_el_tf_energy, my_data_size * sizeof(REAL)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_el_kinetic_energy, my_data_size * sizeof(REAL)));

    energies_struct rank_energies;
    rank_energies.ion_local = 0.0;

    for (int i = 0; i < my_data_size; i++){
        rho_R[i] = {1.0, 0.0};
    }
    
    rho_R[0] = {32, 0};
    rho_R[1] = {64, 0};

    // for(int i = 0; i< my_data_size; i++){
    //     printf("rho(i): %f\n", rho_R[i]);
    // }

    // Setup FFT plan
    unsigned int cufft_plan = setup_fft(nx, ny, nz, MPI_COMM_WORLD);

    // Create GPU descriptors for rho and psi
    cudaLibXtDesc *rho_R_desc, *psi_R_desc, *rho_G_desc, *psi_G_desc;
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &rho_R_desc, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &psi_R_desc, CUFFT_XT_FORMAT_INPLACE));

    // Create GPU descriptor for psi_G, since cudafftxtmemcpy doesnt work directly.
    // I use cudaMemcpy to circumvent this by copying the underlying data after intitializing. TEST
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &psi_G_desc, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &rho_G_desc, CUFFT_XT_FORMAT_INPLACE));

    // Allocate memory for k vectors on CPU and GPU
    REAL *kx, *ky, *kz;
    kx = (REAL*)malloc(sizeof(REAL) * nx);
    ky = (REAL*)malloc(sizeof(REAL) * ny);
    kz = (REAL*)malloc(sizeof(REAL) * nz);

    cudaLibXtDesc *desc_kx, *desc_ky, *desc_kz;
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &desc_kx, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &desc_ky, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMalloc(cufft_plan, &desc_kz, CUFFT_XT_FORMAT_INPLACE));

    // Generate k vectors
    gen_ks(nx, ny, nz,Lx, Ly, Lz, kx, ky, kz, desc_kx, desc_ky, desc_kz, cufft_plan);       

    // Apply G_cut and get normalized density and psi on GPU
    calc_norm_and_psi(rho_R, psi_R, rho_R_desc, rho_G_desc, psi_R_desc, psi_G_desc, my_data_size, nelec, dx ,dy, dz, 
                        cufft_plan, rank, size, nx, ny, nz, desc_kx, desc_ky, desc_kz, cutoff, MPI_COMM_WORLD);


    ///////////////////////////////////
    // SETUP ABOVE CUTOFF
    // NEEDS TO BE MOVED OUTSIDE FUNCTION CALL


    // At this point, we have GPU descriptors for normalized psi_R, psi_G, rho_R, and rho_G, with rho_G CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    // Declare output space, TODO clean and put in a struct (potentials % energies)
}

// THIS FUNCTION IS ONLY CALLED ONCE COULD BE INCLUDED IN INITIALIZE CODE
unsigned int setup_fft(size_t nx, size_t ny, size_t nz, MPI_Comm comm)
{
    // Make cufft plan and set stream
    cufftHandle plan = 0;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftMpAttachComm(plan, CUFFT_COMM_MPI, &comm));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    size_t workspace;
    CUFFT_CHECK(cufftMakePlan3d(plan, nx, ny, nz, CUFFT_Z2Z, &workspace));

    return plan;
}

void finalize(extpot, rho, psi, descx, descy, descz, unsigned int plan)
{
    // // Need to sum potentials on each rank and multiply by psi to get H*psi and calculate mu 

    // Clean up GPU memory
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaFree(d_hartree_pot));
    // CUDA_CHECK(cudaFree(d_xc_pot));
    CUDA_CHECK(cudaFree(d_ext_pot));
    // CUDA_CHECK(cudaFree(d_vw_pot));
    // CUDA_CHECK(cudaFree(d_tf_pot));
    // CUDA_CHECK(cudaFree(d_el_hartree_energy));
    // CUDA_CHECK(cudaFree(d_el_xc_energy));
    // CUDA_CHECK(cudaFree(d_el_tf_energy));
    // CUDA_CHECK(cudaFree(d_el_kinetic_energy));

    CUFFT_CHECK(cufftXtFree(rho_R_desc));
    CUFFT_CHECK(cufftXtFree(psi_R_desc));
    CUFFT_CHECK(cufftXtFree(desc_kx));
    CUFFT_CHECK(cufftXtFree(desc_ky));
    CUFFT_CHECK(cufftXtFree(desc_kz));
    CUFFT_CHECK(cufftDestroy(cufft_plan));

    MPI_Finalize();
}

///////////////////////////////////////////////////////////////////////////
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
///////////////////////////////////////////////////////////////////////////

////////////////////////////////////////
// MAIN FUNCTIONS
////////////////////////////////////////

REAL energy calc_value()
{
    // Calculate Potentials and Energies
    calc_hartree(my_data_size, nx, ny, nz, rho_R_desc, rho_G_desc, d_hartree_pot, d_el_hartree_energy, desc_kx, 
                    desc_ky, desc_kz, rank, size, MPI_COMM_WORLD, cufft_plan, dx, dy, dz);

    thrust::device_ptr<REAL> thrust_el_hartree_energy = thrust::device_pointer_cast(d_el_hartree_energy);
    rank_energies.hartree = thrust::reduce(&thrust_el_hartree_energy[0], &thrust_el_hartree_energy[0] + my_data_size, 0.0f, thrust::plus<REAL>());
    printf("Rank Hartree: %f\n", rank_energies.hartree);
    CUDA_CHECK(cudaFree(d_el_hartree_energy));
    CUDA_CHECK(cudaFree(d_hartree_pot));

    calc_xc(my_data_size, (COMPLEX *)rho_R_desc->descriptor->data[0], d_xc_pot, d_el_xc_energy, rank, dx, dy ,dz, c_light);
    thrust::device_ptr<REAL> thrust_el_xc_energy = thrust::device_pointer_cast(d_el_xc_energy);
    rank_energies.xc = thrust::reduce(&thrust_el_xc_energy[0], &thrust_el_xc_energy[0] + my_data_size, 0.0f, thrust::plus<REAL>());
    printf("Rank xc: %f\n", rank_energies.xc); //negative?
    CUDA_CHECK(cudaFree(d_xc_pot));
    CUDA_CHECK(cudaFree(d_el_xc_energy));

    calc_thomas_fermi(my_data_size, (COMPLEX *)rho_R_desc->descriptor->data[0], d_tf_pot, d_el_tf_energy, rank, size, MPI_COMM_WORLD, ndevices, dx, dy, dz, system_temperature, density_ns, tf_gamma);
    thrust::device_ptr<REAL> thrust_el_tf_energy = thrust::device_pointer_cast(d_el_tf_energy);
    rank_energies.kinetic_local  = thrust::reduce(&thrust_el_tf_energy[0], &thrust_el_tf_energy[0] + my_data_size, 0.0f, thrust::plus<REAL>());
    printf("Rank local KE: %f\n", rank_energies.kinetic_local); //negative?
    CUDA_CHECK(cudaFree(d_tf_pot));
    CUDA_CHECK(cudaFree(d_el_tf_energy));

    if(vW) {

        calc_vw(my_data_size, nx, ny, nz, psi_G_desc, d_vw_pot, d_el_kinetic_energy, desc_kx, 
                    desc_ky, desc_kz, rank, size, MPI_COMM_WORLD, cufft_plan, Lx, Ly, Lz, cutoff, tf_lambda);
                    
        thrust::device_ptr<REAL> thrust_el_kinetic_energy = thrust::device_pointer_cast(d_el_kinetic_energy);
        rank_energies.kinetic = thrust::reduce(&thrust_el_kinetic_energy[0], &thrust_el_kinetic_energy[0] + my_data_size, 0.0f, thrust::plus<REAL>());
        printf("Rank KE: %f\n", rank_energies.kinetic);
        CUDA_CHECK(cudaFree(d_vw_pot));
        CUDA_CHECK(cudaFree(d_el_kinetic_energy));
    }

    rank_energies.total = rank_energies.hartree + rank_energies.ion_local + rank_energies.xc + rank_energies.kinetic_local + rank_energies.kinetic;
    printf("Rank Total Energy: %f\n", rank_energies.total);

    REAL total_energy;
    // Perform the MPI_Allreduce operation to compute the sum across all ranks
    MPI_Allreduce(&rank_energies.total, &total_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    printf("Global Total Energy: %f\n", total_energy);
}

//REAL* calc_derivative = ()
//{
//    if(vW) 
//    {
//        //potentials%kinetic_local%of(i)%R= potentials%kinetic_local%of(i)%R - density%n_s*tf%lambda*d2psi%of(i)%R/(2.0_dp*psi%of(i)%R)
//    }   //                                  " ^^^^ this is d_tf_pot"                                  "d_vw_pot is 1/2 * d2psi_R"
//        // Wrap this up in a kernel to calculate kinetic local then add then multiply by psi elementwise to get H*psi 
//}


////////////////////////////////////////////////////////////////////////////////
// FUNCTIONS TO CALCULATE POTENTIALS FOR HAMILTONIAN
////////////////////////////////////////////////////////////////////////////////

// 1) Thomas Fermi (Local Kinetic energy)
// 2) Von Wiesacker (Kinetic energy correction)
// 3) Pseudopotential (electron to atomic nucleus electric field)
// 4) Hartree (electron to electron electric field)
// 5) Exchange Correlation (LDA)
// 6) Exchange Correlation with Von Wiesacker

// ------------------------------------------------------

// 1)
void calc_thomas_fermi(size_t my_data_size, COMPLEX *d_rho_R, REAL *d_tf_pot, REAL *d_el_local_kinetic_energy,
                        int rank, int size, MPI_Comm comm, int ndevices, REAL dx, REAL dy, REAL dz, REAL system_temperature, int density_ns,  REAL tf_gamma)
{
    // Define CUDA block and grid dimensions
    int num_threads = 1024;
    int num_blocks = (my_data_size + num_threads - 1) / num_threads;

    // Process the data on the GPU
    tf_pot_kernel<<<num_blocks, num_threads>>>(d_rho_R, d_tf_pot, d_el_local_kinetic_energy, my_data_size, system_temperature, 
                                                dx, dy, dz, density_ns, tf_gamma);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
}

// 2) Von Wiesacker
void calc_vw(size_t my_data_size, size_t nx, size_t ny, size_t nz, cudaLibXtDesc *psi_G_desc, REAL *d_vw_pot, REAL *d_el_kinetic_energy,
                cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, int rank, int size, MPI_Comm comm, unsigned int plan,
                 REAL Lx, REAL Ly, REAL Lz, REAL cutoff, REAL tf_lambda)
{
    cudaLibXtDesc *vw_desc;

    CUFFT_CHECK(cufftXtMalloc(plan, &vw_desc, CUFFT_XT_FORMAT_INPLACE));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy((void *)vw_desc->descriptor->data[0], (void *)psi_G_desc->descriptor->data[0], my_data_size * sizeof(COMPLEX), cudaMemcpyDeviceToDevice)); 

    auto [vw_begin_d, vw_end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                         rank, size, nx, ny, nz, (COMPLEX *)vw_desc->descriptor->data[0]);

    const size_t num_threads = 1024; // 1024 for A100 GPU
    const size_t num_blocks = (my_data_size + num_threads - 1) / num_threads;

    // psi_G is already scaled
    vw_kernel<<<num_blocks, num_threads>>>(vw_begin_d, vw_end_d, rank, size, nx, ny, nz,
                                                         (REAL *)desc_kx->descriptor->data[0], (REAL *)desc_ky->descriptor->data[0], 
                                                         (REAL *)desc_kz->descriptor->data[0], d_el_kinetic_energy, Lx ,Ly, Lz, cutoff, tf_lambda);

    // Run FFT backwards to get vw potential
    CUFFT_CHECK(cufftXtExecDescriptor(plan, vw_desc, vw_desc, CUFFT_INVERSE));
    // Data is distributed as X-Slabs again
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    cudaMemcpy((void *)d_vw_pot, (void *)vw_desc->descriptor->data[0], my_data_size * sizeof(REAL), cudaMemcpyDeviceToDevice); // 1/2 * d2psi_R

    // Free memory
    CUFFT_CHECK(cufftXtFree(vw_desc));
}

// 3) PSEUDOPOTENTIAL (ATOIMC NUCLEUS ELECTRIC FIELDS)
void calc_local_pp_energy(size_t my_data_size, COMPLEX d_rho_R, REAL *local_kinetic_pot,
                        int rank, int size, MPI_Comm comm, int ndevices)
{

//         if(any(density%of(:)%grid.ne.local_PP%of%grid)) then
//             print *, 'Attempting to calculate Local Ion energy from density and potential on different grid, stopping'
//             stop
//         endif

//         Een_local= 0.0_dp
//         do i=1, density%n_s
//             Een_local= Een_local + real(integrate_3D_R( &
//                 real(density%of(i)%R)*real(local_PP%of%R), grids(local_PP%of%grid), parallel))
//         enddo
}

// 4) ELECTRON TO ELECTRON ELECTRIC FIELD
void calc_hartree(size_t my_data_size, size_t nx, size_t ny, size_t nz, cudaLibXtDesc *rho_R_desc, cudaLibXtDesc *rho_G_desc, REAL *d_hartree_pot, REAL *d_el_hartree_energy,
                cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, int rank, int size, MPI_Comm comm, unsigned int plan, REAL dx, REAL dy, REAL dz)
{
    cudaLibXtDesc *hartree_desc;

    CUFFT_CHECK(cufftXtMalloc(plan, &hartree_desc, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy((void *)hartree_desc->descriptor->data[0], (void *)rho_G_desc->descriptor->data[0], my_data_size * sizeof(COMPLEX), cudaMemcpyDeviceToDevice)); 

    auto [hartree_begin_d, hartree_end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                         rank, size, nx, ny, nz, (COMPLEX *)hartree_desc->descriptor->data[0]);

    const size_t num_threads = 1024; // 1024 for A100 GPU
    const size_t num_blocks = (my_data_size + num_threads - 1) / num_threads;

    // rho_G is already scaled
    hartree_kernel<<<num_blocks, num_threads>>>(hartree_begin_d, hartree_end_d, rank, size,
                                                        (REAL *)desc_kx->descriptor->data[0], (REAL *)desc_ky->descriptor->data[0], 
                                                        (REAL *)desc_kz->descriptor->data[0]);

    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    // Run FFT backwards to get Hartree potential
    CUFFT_CHECK(cufftXtExecDescriptor(plan, hartree_desc, hartree_desc, CUFFT_INVERSE));

    // Calculate energy and return
    hartree_energy_kernel<<<num_blocks, num_threads>>>((COMPLEX *)rho_R_desc->descriptor->data[0], (COMPLEX *)hartree_desc->descriptor->data[0], 
                                                d_hartree_pot, d_el_hartree_energy, my_data_size, dx, dy, dz);    

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUFFT_CHECK(cufftXtFree(hartree_desc));
}

// 5) EXCHANGE CORRELATION (LDA)
void calc_xc_vwn(size_t my_data_size, COMPLEX *d_rho_R, REAL *d_xc_pot, REAL *d_el_xc_energy, int rank, REAL dx, REAL dy, REAL dz, REAL c_light)
{
    REAL y0 = -0.10498;
    REAL b = 3.72744;
    REAL c = 12.9352;
    REAL A = 0.0621814;
    bool relat = false;
    
    REAL *d_xc_den;
    CUDA_CHECK_ERROR(cudaMalloc(&d_xc_den, my_data_size * sizeof(REAL)));

    // Define CUDA block and grid dimensions
    int num_threads = 1024;
    int num_blocks = (my_data_size + num_threads - 1) / num_threads;

    // Process the data on the GPU
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    calc_xc_vwn<<<num_blocks, num_threads>>>(d_rho_R, d_xc_pot, d_xc_den, d_el_xc_energy, my_data_size, dx, dy, dz,
                                            y0, b, c, A, relat, c_light);
}


////////////////////////////////////////////////////////////////////////////////
// KERNELS FOR POTENTIAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__global__ void tf_pot_kernel(COMPLEX *d_rho_R, REAL *d_local_kinetic_pot, REAL *d_elemental_local_kinetic_energy, int data_size, REAL system_temperature, 
                            REAL dx, REAL dy, REAL dz, int density_ns, REAL tf_gamma)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting from line 133 in Thomas_Fermi.f90:
    REAL dy0dn = pow(M_PI, 2) / sqrt(2.0) * pow(system_temperature, -3.0/2.0);

    if (tid_x < data_size) {
        // Thomas fermi potential (line 135 in Thomas_Fermi.f90)
        REAL d_density_R = d_rho_R[tid_x].x; // psi^2
        
        d_local_kinetic_pot[tid_x] = system_temperature * (elem(dy0dn * d_density_R * density_ns)
             + d_density_R * density_ns * dy0dn * elem(dy0dn * d_density_R * density_ns, true, false));
        
        d_local_kinetic_pot[tid_x]  *= tf_gamma;
        
        d_elemental_local_kinetic_energy[tid_x] = system_temperature * d_density_R * density_ns * elem(dy0dn * d_density_R * density_ns);
        d_elemental_local_kinetic_energy[tid_x] = tf_gamma * d_elemental_local_kinetic_energy[tid_x] / density_ns * dx * dy * dz;

        // if (tid_x < 10){
        //     printf("tid_x: %d, local data: %f\n", tid_x, d_elemental_local_kinetic_energy[tid_x]);
        // }
    }
}

__global__ void vw_kernel(BoxIterator<COMPLEX> d_psi_G_begin, BoxIterator<COMPLEX> d_psi_G_end, int rank, int size, size_t nx, size_t ny, size_t nz,
                            REAL *kx, REAL *ky, REAL *kz, REAL *d_kinetic_energy, REAL Lx, REAL Ly, REAL Lz, REAL cutoff, REAL tf_lambda)  // Calculates VW potential and energy pointwise
{
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x; 
    d_psi_G_begin += tid_x;

    if (d_psi_G_begin < d_psi_G_end)
    {
        // Compute g^2 and apply. Cutoff is already applied.
        REAL g2 = abs(kx[(int)d_psi_G_begin.x()]*kx[(int)d_psi_G_begin.x()] + ky[(int)d_psi_G_begin.y()]*ky[(int)d_psi_G_begin.y()] + kz[(int)d_psi_G_begin.z()]*kz[(int)d_psi_G_begin.z()]);

        // Line 88 in Orbital_Free_Min.f90
        d_kinetic_energy[tid_x] = tf_lambda * ((d_psi_G_begin->x * d_psi_G_begin->x) + (d_psi_G_begin->y * d_psi_G_begin->y)) * g2 * 0.5 * Lx * Ly * Lz; // lambda * abs(psi_G)^2 * g2 / 2 * Lx * Ly * Lz
        *d_psi_G_begin = {d_psi_G_begin->x * g2 * 0.5, d_psi_G_begin->y * g2 * 0.5}; // 1/2 d2psi_G

        // if (tid_x < 10) {
        //     printf("tid: %d, KE kernel value: %f\n",  tid_x, d_kinetic_energy[tid_x]);
        // }
    }
}

__global__ void hartree_kernel(BoxIterator<COMPLEX> d_rho_G_begin, BoxIterator<COMPLEX> d_rho_G_end, int rank, int size, REAL *kx, REAL *ky, REAL *kz) // Returns 4*pi*psi_G/g^2
{ // Calculates VeeG = 4*pi*neG / G2, but skips G2=0 point
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x; 
    d_rho_G_begin += tid_x;

    if (d_rho_G_begin < d_rho_G_end)
    {
        // d_psi_G_begin.x(), d_psi_G_begin.y() and d_psi_G_begin.z() are the global 3D coordinate of the data pointed by the iterator
        // d_psi_G_begin->x and d_psi_G_begin->y are the real and imaginary part of the corresponding COMPLEX element
        // if (tid_x < 10) 
        // {
        //     printf("GPU data (before hartree): global 3D index [%d %d %d], local index %d, rank %d is (%f,%f). kx: %f, ky: %f, kz: %f\n",
        //            (int)d_rho_G_begin.x(), (int)d_rho_G_begin.y(), (int)d_rho_G_begin.z(), (int)d_rho_G_begin.i(), rank, d_rho_G_begin->x, d_rho_G_begin->y, kx[(int)d_rho_G_begin.x()], ky[(int)d_rho_G_begin.y()], kz[(int)d_rho_G_begin.z()]);
        // }

        REAL g2 = abs(kx[(int)d_rho_G_begin.x()]*kx[(int)d_rho_G_begin.x()] + ky[(int)d_rho_G_begin.y()]*ky[(int)d_rho_G_begin.y()] + kz[(int)d_rho_G_begin.z()]*kz[(int)d_rho_G_begin.z()]);
        
        // Line 32 Hartree.f90
        // Cutoff has already been applied
        if (g2 < DBL_MIN) 
        {
            *d_rho_G_begin = {0.0, 0.0}; // Charge neutrality condition
            return;
        }
        *d_rho_G_begin = {d_rho_G_begin->x * 4.0 * M_PI / g2, d_rho_G_begin->y * 4.0 * M_PI / g2}; // 4 * pi * abs(psi_G)^2 / g^2
    }
}

__global__ void hartree_energy_kernel(COMPLEX *d_rho_R, COMPLEX *d_hartree_pot_desc, REAL *d_hartree_pot, REAL *d_elemental_hartree_energy, int data_size, REAL dx, REAL dy, REAL dz)
{
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x; 

    if (tid_x < data_size) {
    {   
        // Line 65 in Hartree.f90
        d_hartree_pot[tid_x] = d_hartree_pot_desc[tid_x].x;
        d_elemental_hartree_energy[tid_x] = d_hartree_pot[tid_x] * d_rho_R[tid_x].x / 2.0 * dx * dy *dz; //real(integrate_3D_R(real(density%of(i)%R)*real(hartree%of%R), grid, parallel))/2.0_dp
    }
}
}

__global__ void calc_xc_vwn_kernel(COMPLEX *d_rho_R, REAL *d_vxc, REAL *d_exc, REAL *d_elemental_xc_energy, size_t my_data_size,  
                REAL dx, REAL dy, REAL dz, REAL y0, REAL b, REAL c, REAL A, bool relat, REAL c_light)
{
    // n = charge density (scalar)
    // relat ! if .true. returns RLDA, otherwise LDA
    // exc ! XC density
    // Vxc ! XC potential

    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    REAL Q, rs, y, ec, ex, Vc, Vx, beta, mu, R, S;
        
    // XC pot from xc_pz2. Line 168 in xc.f90
    if (tid_x < my_data_size) {

        if(abs(d_rho_R[tid_x].x) < DBL_MIN) {
            d_exc[tid_x] = 0.0;
            d_vxc[tid_x] = 0.0; 
            return;
        }

        Q = sqrt(4 * c - b*b);
        rs = (3 / pow(4 * M_PI * d_rho_R[tid_x].x, 1.0/3.0));
        y = sqrt(rs);

        ec = A / 2 * (log(y*y / get_Y(y,b,c)) + 2*b/Q * atan(Q/(2*y+b)) - b*y0/get_Y(y0,b,c)
                        * (log((y-y0)*(y-y0)/get_Y(y,b,c)) + 2*(b+2*y0)/Q*atan(Q/(2*y+b)))); // double check
        Vc = ec - A/6 * (c*(y-y0)-b*y0*y)/((y-y0)*get_Y(y, b, c));
        ex = -3/(4 * M_PI) * pow(3 * M_PI * M_PI * d_rho_R[tid_x].x, 1.0/3.0);
        Vx = 4*ex/3;

        if(relat == true){
            beta = -4 * M_PI * ex / (3 * c_light);
            mu = sqrt(1 + beta*beta);
            R = 1 - 3 * pow((beta * mu - log(beta + mu)) / (beta * beta), 2) / 2;
            S = 3 * log(beta + mu) / (2 * beta * mu) - 1.0 / 2.0;
        
            ex = ex * R;
            Vx = Vx * S;
        }

        d_exc[tid_x] = ex + ec;
        d_vxc[tid_x]= Vx + Vc;

        d_elemental_xc_energy[tid_x] = d_exc[tid_x] * d_rho_R[tid_x].x * dx * dy * dz; //line 216 in SHRED XC.f90
        // if (tid_x < 10){
        //     printf("tid_x: %d, local data: %f\n", tid_x, d_elemental_local_kinetic_energy[tid_x]);
        // }
    }
}


////////////////////////////////////////////////////////////////////////////////
// OTHER KERNELS
////////////////////////////////////////////////////////////////////////////////

__global__ void scaling_kernel(BoxIterator<COMPLEX> d_begin, BoxIterator<COMPLEX> d_end, int rank, int size, size_t nx, size_t ny, size_t nz)
{
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    d_begin += tid_x;
    if (d_begin < d_end)
    {
        *d_begin = {d_begin->x / (REAL)(nx * ny * nz), d_begin->y / (REAL)(nx * ny * nz)};

        // if (tid_x < 10) 
        // {
        //     printf("GPU data (after scaling): global 3D index [%d %d %d], local index %d, rank %d is (%f,%f).\n",
        //            (int)d_begin.x(), (int)d_begin.y(), (int)d_begin.z(), (int)d_begin.i(), rank, d_begin->x, d_begin->y);
        // }
    }
}


////////////////////////////////////////////////////////////////////////////////
// SUPPORTING FUNCTIONS FOR KERNELS
////////////////////////////////////////////////////////////////////////////////


// Thomas Fermi Calls this Specially Developed Parametric Fit Model
__device__ REAL elem(REAL y, bool deriv = false, bool sec_deriv = false)
{   
    REAL f = 0.0, u = 0.0;
    REAL dudy, d2udy2;
    REAL y0 = 3 * M_PI  / (4 * sqrt(2.0));

    REAL c[8] = {-0.8791880215, 0.1989718742, 0.1068697043e-2, -0.8812685726e-2, 0.1272183027e-1,
                -0.9772758583e-2, 0.3820630477e-2, -0.5971217041e-3};

    REAL d[9] = {0.7862224183, -0.1882979454e1, 0.5321952681, 0.2304457955e1, -0.1614280772e2,
                0.5228431386e2, -0.9592645619e2, 0.9462230172e2, -0.3893753937e2};


    if(deriv != true) {       
        if(sec_deriv != true) {
            // Desire f(y)
            if (y <= y0) {
                f = log(y);
                for (int i = 0; i <= 7; ++i) {
                    f += c[i] * pow(y,i);
                }

            } else {
                u = pow(y, 2.0/3.0);
                f = 0;

                for (int i = 0; i <= 8; ++i) {
                    f += d[i] / pow(u, 2*i - 1);
                }
            } 
        } else {
            // Desire df(y)/dy

            if (y <= y0) {
                f = 1 / y;
                for (int i = 1; i <= 7; ++i) {
                    f += (i) * c[i] * pow(y, i-1);
                }
            } else {
                u = pow(y, 2.0/3.0);
                dudy = 2.0 / 3.0 / pow(y, 1.0/3.0);
                f = 0;
                for (int i = 0; i <= 8; ++i) {
                    f += (1 - 2*i) * d[i] / pow(u, 2*i);
                }
            }
        } // End if(sec_deriv != true)
        
        
    } else {
        // Desire d^2f(y)/dy^2
        if (y <= y0) {
            f = -1.0 / pow(y, 2);
            for (int i = 2; i <= 7; ++i) {
                f += i * (i-1) * c[i] * pow(y, i-2);
            }
        } else {
            u = pow(y, 2.0/3.0);
            dudy = 2.0 / 3.0 / pow(y, 1.0/3.0);
            dudy = pow(dudy, 2);
            d2udy2 = -2.0/9.0 / pow(y, 4.0 / 3.0);
            f = 0;
            for (int i = 1; i <= 8; ++i) {
                f -= dudy * (1 - 2*i) * (2*i) * d[i] / pow(u, 2*i + 1);
            }
            for (int i = 0; i <= 8; ++i) {
                f += d2udy2 * (1 - 2*i) * (2*i) * d[i] / pow(u, 2*i);
            }
        }
    } // End if(deriv != true)

    return f;
}


////////////////////////////////////////
// WAVEFUNCTION, DENSITY, NORM, CUTOFF
////////////////////////////////////////

__global__ void calc_psi(COMPLEX *d_rho_R, COMPLEX *d_psi_R, REAL *d_psi_R_sq_dr, size_t my_data_size, REAL dx, REAL dy, REAL dz)
{
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting from line 191 in Orbital_Free_Min.f90:
    if (tid_x < my_data_size) {
        d_psi_R[tid_x].x = sqrt(d_rho_R[tid_x].x);
        d_psi_R_sq_dr[tid_x] = d_psi_R[tid_x].x * d_psi_R[tid_x].x * dx * dy *dz;

        // if (tid_x < 100) {
        //     printf("tid: %d, rho.x: %f\n",  tid_x, d_rho_R[tid_x].x);
        // }
    }
}

__global__ void calc_density(COMPLEX *d_psi_R, COMPLEX *d_rho_R, size_t my_data_size)
{
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting from line 191 in Orbital_Free_Min.f90:
    if (tid_x < my_data_size) {
        d_rho_R[tid_x].x = d_psi_R[tid_x].x * d_psi_R[tid_x].x;

        // if (tid_x < 10) {
        //     printf("tid: %d, density: %f\n",  tid_x, d_rho_R[tid_x].x);
        // }
    }

}

__global__ void calc_norm1(COMPLEX *d_psi_R, REAL *d_psi_R_sq_dr, size_t my_data_size, REAL dx, REAL dy, REAL dz)
{
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting from line 191 in Orbital_Free_Min.f90:
    if (tid_x < my_data_size) {
        d_psi_R_sq_dr[tid_x] = d_psi_R[tid_x].x * d_psi_R[tid_x].x * dx * dy *dz;
    }

    // if (tid_x < 10) {
    //     printf("tid: %d, psi: (%f, %f), d_psi_R_sq_dr: %f\n",  tid_x, d_psi_R[tid_x].x, d_psi_R[tid_x].y, d_psi_R_sq_dr[tid_x]);
    // }
}

__global__ void calc_norm2(COMPLEX *d_psi_R, COMPLEX *d_psi_G, REAL *d_psi_R_sq_dr, REAL psi_norm_before, size_t my_data_size, size_t nelec, REAL dx, REAL dy, REAL dz)
{
    const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting from line 191 in Orbital_Free_Min.f90:
    if (tid_x < my_data_size) {

        d_psi_R[tid_x].x = d_psi_R[tid_x].x * sqrt(nelec / psi_norm_before);
        d_psi_G[tid_x].x = d_psi_G[tid_x].x * sqrt(nelec / psi_norm_before);

        // Prepare integral second norm calculation
        d_psi_R_sq_dr[tid_x] = d_psi_R[tid_x].x * d_psi_R[tid_x].x * dx * dy *dz;

        // if (tid_x < 10) {
        //     printf("tid: %d, psi_G: (%f, %f) \n",  tid_x, d_psi_G[tid_x].x, d_psi_G[tid_x].y);
        // }
    }

}

__global__ void apply_cutoff(BoxIterator<COMPLEX> d_psi_G_begin, BoxIterator<COMPLEX> d_psi_G_end, REAL *kx, REAL *ky, REAL *kz, REAL cutoff)
{
    const int tid_x = threadIdx.x + blockIdx.x * blockDim.x; 
    d_psi_G_begin += tid_x;

    if (d_psi_G_begin < d_psi_G_end)
    {
        // d_psi_G_begin.x(), d_psi_G_begin.y() and d_psi_G_begin.z() are the global 3D coordinate of the data pointed by the iterator
        // d_psi_G_begin->x and d_psi_G_begin->y are the real and imaginary part of the corresponding COMPLEX element

        // Compute g^2 and apply within the cutoff radius
        REAL g2 = abs(kx[(int)d_psi_G_begin.x()]*kx[(int)d_psi_G_begin.x()] + ky[(int)d_psi_G_begin.y()]*ky[(int)d_psi_G_begin.y()] + kz[(int)d_psi_G_begin.z()]*kz[(int)d_psi_G_begin.z()]);

        if (0.5 * g2 > cutoff) 
        {
            *d_psi_G_begin = {0.0, 0.0};
            return;
        }
    
        // if (tid_x < 10) 
        //     {
        //         printf("GPU data (before hartree): global 3D index [%d %d %d], local index %d is (%f,%f). kx: %f, ky: %f, kz: %f\n",
        //             (int)d_psi_G_begin.x(), (int)d_psi_G_begin.y(), (int)d_psi_G_begin.z(), (int)d_psi_G_begin.i(), d_psi_G_begin->x, d_psi_G_begin->y, kx[(int)d_psi_G_begin.x()], ky[(int)d_psi_G_begin.y()], kz[(int)d_psi_G_begin.z()]);
        //     }
    }
}

void calc_norm_and_psi(COMPLEX *rho_R, COMPLEX *psi_R, cudaLibXtDesc *rho_R_desc, cudaLibXtDesc *rho_G_desc, cudaLibXtDesc *psi_R_desc, cudaLibXtDesc *psi_G_desc, size_t my_data_size, size_t nelec, REAL dx, REAL dy, REAL dz, 
                    unsigned int plan, int rank, int size, size_t nx, size_t ny, size_t nz, cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, REAL cutoff, MPI_Comm comm)
{
            // psi%of(i)%R = sqrt(density%of(i)%R)  
            // call real_to_recip(psi%of(i), grids) 
            // psi%of(i)%G=psi%of(i)%G*grid%cutwf
            // call recip_to_real(psi%of(i), grids) 
            // psi_norm = integrate_3D_R( abs(psi%of(i)%R)**2, grid, parallel)
            // psi%of(i)%R = psi%of(i)%R * sqrt( system%nelec(i) / psi_norm) 
            // psi%of(i)%G = psi%of(i)%G * sqrt( system%nelec(i) / psi_norm) not done here
            // psi_norm = integrate_3D_R( abs(psi%of(i)%R)**2, grid, parallel)
            // density%of(i)%R = abs(psi%of(i)%R)**2
            // call real_to_recip(density%of(i), grids)

    REAL *d_psi_R_sq_dr;
    CUDA_CHECK_ERROR(cudaMalloc(&d_psi_R_sq_dr, my_data_size * sizeof(REAL)));

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUFFT_CHECK(cufftXtMemcpy(plan, (void *)rho_R_desc, (void *)rho_R, CUFFT_COPY_HOST_TO_DEVICE)); // Copy rho_R to hartee to keep rho_R intact

    // Define CUDA block and grid dimensions
    int num_threads = 1024;
    int num_blocks = (my_data_size + num_threads - 1) / num_threads;

    // Calculate Psi and Psi_sq on the GPU
    calc_psi<<<num_blocks, num_threads>>>((COMPLEX *)rho_R_desc->descriptor->data[0], (COMPLEX *)psi_R_desc->descriptor->data[0], d_psi_R_sq_dr, my_data_size, dx, dy, dz);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Run FFT forward, data is now CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    CUFFT_CHECK(cufftXtExecDescriptor(plan, psi_R_desc, psi_R_desc, CUFFT_FORWARD));

    auto [psi_R_begin_d, psi_R_end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                         rank, size, nx, ny, nz, (COMPLEX *)psi_R_desc->descriptor->data[0]);

    
    scaling_kernel<<<num_blocks, num_threads>>>(psi_R_begin_d, psi_R_end_d, rank, size, nx, ny, nz);
    apply_cutoff<<<num_blocks, num_threads>>>(psi_R_begin_d, psi_R_end_d, (REAL *)desc_kx->descriptor->data[0], 
                                                (REAL *)desc_ky->descriptor->data[0], (REAL *)desc_kz->descriptor->data[0],cutoff);

    // Create GPU descriptor for psi_G, since cudafftxtmemcpy doesnt work directly.
    // I use cudaMemcpy to circumvent this by copying the underlying data after intitializing. TEST
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy((void *)psi_G_desc->descriptor->data[0], (void *)psi_R_desc->descriptor->data[0], my_data_size * sizeof(COMPLEX), cudaMemcpyDeviceToDevice)); 

    // Run IFFT, data is now CUFFT_XT_FORMAT_INPLACE again
    CUFFT_CHECK(cufftXtExecDescriptor(plan, psi_R_desc, psi_R_desc, CUFFT_INVERSE));
    
    // First normalization, doesnt change psi
    calc_norm1<<<num_blocks, num_threads>>>((COMPLEX *)psi_R_desc->descriptor->data[0], d_psi_R_sq_dr, my_data_size, dx, dy, dz);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Calculate norm
    thrust::device_ptr<REAL> thrust_d_psi_R_sq1 = thrust::device_pointer_cast(d_psi_R_sq_dr);
    REAL psi_norm_before = thrust::reduce(&thrust_d_psi_R_sq1[0], &thrust_d_psi_R_sq1[0] + my_data_size, 0.0f, thrust::plus<REAL>());
    printf("First norm: %f\n", psi_norm_before);

    // MPI_Allreduce to compute the the total norm across all ranks
    REAL total_norm1;
    MPI_Allreduce(&psi_norm_before, &total_norm1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Normalize psi_R and psi_Gand calculate norm again
    calc_norm2<<<num_blocks, num_threads>>>((COMPLEX *)psi_R_desc->descriptor->data[0], (COMPLEX *)psi_G_desc->descriptor->data[0], d_psi_R_sq_dr, total_norm1, my_data_size, nelec, dx, dy, dz);
    thrust::device_ptr<REAL> thrust_d_psi_R_sq2 = thrust::device_pointer_cast(d_psi_R_sq_dr);
    REAL psi_norm_after = thrust::reduce(&thrust_d_psi_R_sq2[0], &thrust_d_psi_R_sq2[0] + my_data_size, 0.0f, thrust::plus<REAL>());
    printf("Second norm: %f\n", psi_norm_after);

    REAL total_norm2;
    // Perform the MPI_Allreduce operation to compute the sum across all ranks
    MPI_Allreduce(&psi_norm_after, &total_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    calc_density<<<num_blocks, num_threads>>>((COMPLEX *)psi_R_desc->descriptor->data[0], (COMPLEX *)rho_G_desc->descriptor->data[0], my_data_size);
    // rho_G_desc now holds real R3 data. 

    // Copy data from rho_G_desc to rho_R_desc
    // Create GPU descriptor for rho_R_desc, since cudafftxtmemcpy doesnt work directly.
    // I use cudaMemcpy to circumvent this by copying the underlying data after intitializing. TEST
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy((void *)rho_R_desc->descriptor->data[0], (void *)rho_G_desc->descriptor->data[0], my_data_size * sizeof(COMPLEX), cudaMemcpyDeviceToDevice)); 

    // Run FFT forward  for rho_R, data is now CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    CUFFT_CHECK(cufftXtExecDescriptor(plan, rho_G_desc, rho_G_desc, CUFFT_FORWARD));

    auto [rho_G_begin_d, rho_G_end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_Z2Z,
                                         rank, size, nx, ny, nz, (COMPLEX *)rho_G_desc->descriptor->data[0]);

    scaling_kernel<<<num_blocks, num_threads>>>(rho_G_begin_d, rho_G_end_d, rank, size, nx, ny, nz);

    // At this point, we have GPU descriptors for normalized psi_R, psi_G, rho_R, and rho_G    
}


////////////////////////////////////////////////////////////////////////////////
// MATH FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

int displacement(int length, int rank, int size) {
    int ranks_cutoff = length % size;
    return (rank < ranks_cutoff ? rank * (length / size + 1) : ranks_cutoff * (length / size + 1) + (rank - ranks_cutoff) * (length / size));
}

__device__ REAL get_Y(REAL y, REAL b, REAL c) {

    return y*y + b*y + c;
}


////////////////////////////////////////
// GRID SETUP
////////////////////////////////////////

void gen_ks (size_t nx, size_t ny, size_t nz, REAL Lx, REAL Ly, REAL Lz, REAL *kx, REAL *ky, REAL *kz,
            cudaLibXtDesc *desc_kx, cudaLibXtDesc *desc_ky, cudaLibXtDesc *desc_kz, unsigned int cufft_plan)
{
    // For shuffled in place, make sure that the x's are corectly distributed among ranks/gpus. Will need to offset x index depd_psi_G_ending on rank.
    for (int i=0; i<=nx/2; i++)
    {
        kx[i] = i * 2*M_PI/Lx;
        // printf("%f \n", kx[i]);
    }
    for (int i=nx/2+1; i<nx; i++)
    {
        kx[i] = (i - static_cast<int>(nx)) * 2*M_PI/Lx;
        // printf("%f \n", kx[i]);
    }

    for (int i=0; i<=ny/2; i++)
    {
        ky[i] = i * 2*M_PI/Ly;
        // printf("%f \n", ky[i]);
    }
    for (int i=ny/2+1; i<ny; i++)
    {
        ky[i] = (i - static_cast<int>(ny)) * 2*M_PI/Ly;
        // printf("%f \n", ky[i]);
    }

    for (int i=0; i<=nz/2; i++)
    {
        kz[i] = i * 2*M_PI/Lz;
        // printf("%f \n", kz[i]);
    }
    for (int i=nz/2+1; i<nz; i++)
    {
        kz[i] = (i - static_cast<int>(nz)) * 2*M_PI/Lz;
        // printf("%f \n", kz[i]);
    }

    // Copy k vectors to GPU descriptors
    CUFFT_CHECK(cufftXtMemcpy(cufft_plan, (void *)desc_kx, (void *)kx, CUFFT_COPY_HOST_TO_DEVICE));
    CUFFT_CHECK(cufftXtMemcpy(cufft_plan, (void *)desc_ky, (void *)ky, CUFFT_COPY_HOST_TO_DEVICE));
    CUFFT_CHECK(cufftXtMemcpy(cufft_plan, (void *)desc_kz, (void *)kz, CUFFT_COPY_HOST_TO_DEVICE));
}


////////////////////////////////////////////////////////////////////////////////
// PRINT FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void printdoubleArray(REAL* arr, int size) {
    if (arr == nullptr || size <= 0) {
        std::cout << "Invalid array or size." << std::endl;
        return;
    }

    std::cout << "double Array: ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}