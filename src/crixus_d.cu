#ifndef CRIXUS_D_CU
#define CRIXUS_D_CU

#include <cuda.h>
#include "lock.cuh"
#include "crixus_d.cuh"
#include "return.h"
#include "crixus.h"

__global__ void set_bound_elem (uf4 *pos, uf4 *norm, float *surf, ui4 *ep, unsigned int nbe, float *xminp, float *xminn, float *nminp, float*nminn, Lock lock, int nvert)
{
	float ddum[3];
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ float xminp_c[threadsPerBlock];
	__shared__ float xminn_c[threadsPerBlock];
	__shared__ float nminp_c[threadsPerBlock];
	__shared__ float nminn_c[threadsPerBlock];
	float xminp_t;
	float xminn_t;
	float nminp_t;
	float nminn_t;
	int i_c = threadIdx.x;
	xminp_t = *xminp;
	xminn_t = *xminn;
	nminp_t = *nminp;
	nminn_t = *nminn;
	while(i<nbe){
		//formula: a = 1/4 sqrt(4*a^2*b^2-(a^2+b^2-c^2)^2)
		float a2 = 0.;
		float b2 = 0.;
		float c2 = 0.;
		ddum[0] = 0.;
		ddum[1] = 0.;
		ddum[2] = 0.;
		for(unsigned int j=0; j<3; j++){
			ddum[j] += pos[ep[i].a[0]].a[j]/3.;
			ddum[j] += pos[ep[i].a[1]].a[j]/3.;
			ddum[j] += pos[ep[i].a[2]].a[j]/3.;
			a2 += pow(pos[ep[i].a[0]].a[j]-pos[ep[i].a[1]].a[j],2);
			b2 += pow(pos[ep[i].a[1]].a[j]-pos[ep[i].a[2]].a[j],2);
			c2 += pow(pos[ep[i].a[2]].a[j]-pos[ep[i].a[0]].a[j],2);
		}
		if(norm[i].a[2] > 1e-5 && xminp_t > ddum[2]){
			xminp_t = ddum[2];
			nminp_t = norm[i].a[2];
		}
		if(norm[i].a[2] < -1e-5 && xminn_t > ddum[2]){
			xminn_t = ddum[2];
			nminn_t = norm[i].a[2];
		}
		surf[i] = 0.25*sqrt(4.*a2*b2-pow(a2+b2-c2,2));
    for(int j=0; j<3; j++)
		  pos[i+nvert].a[j] = ddum[j];
		i += blockDim.x*gridDim.x;
	}

	xminp_c[i_c] = xminp_t;
	xminn_c[i_c] = xminn_t;
	nminp_c[i_c] = nminp_t;
	nminn_c[i_c] = nminn_t;
	__syncthreads();

	int j = blockDim.x/2;
	while (j!=0){
		if(i_c < j){
			if(xminp_c[i_c+j] < xminp_c[i_c]){
				xminp_c[i_c] = xminp_c[i_c+j];
				nminp_c[i_c] = nminp_c[i_c+j];
			}
			if(xminn_c[i_c+j] < xminn_c[i_c]){
				xminn_c[i_c] = xminn_c[i_c+j];
				nminn_c[i_c] = nminn_c[i_c+j];
			}
		}
		__syncthreads();
		j /= 2;
	}

	if(i_c == 0){
		lock.lock();
		if(xminp_c[0] < *xminp){
			*xminp = xminp_c[0];
			*nminp = nminp_c[0];
		}
		if(xminn_c[0] < *xminn){
			*xminn = xminn_c[0];
			*nminn = nminn_c[0];
		}
		lock.unlock();
	}
}

__global__ void swap_normals (uf4 *norm, int nbe)
{
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	while(i<nbe){
    for(int j=0; j<3; j++)
		  norm[i].a[j] *= -1.;
		i += blockDim.x*gridDim.x;
	}
}

__global__ void periodicity_links (uf4 *pos, ui4 *ep, int nvert, int nbe, uf4 *dmax, uf4 *dmin, float dr, int *sync_i, int *sync_o, int *newlink, int idim, Lock lock)
{
	//find corresponding vertices
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int i_c = threadIdx.x;
	while(i<nvert){
		newlink[i] = 0;
		i += blockDim.x*gridDim.x;
	}

//	gpu_sync(sync_i, sync_o);
	if(i_c==0){
		lock.lock();
		lock.unlock();
	}
	__syncthreads();

	i = blockIdx.x*blockDim.x+threadIdx.x;
	while(i<nvert){
		if(fabs(pos[i].a[idim]-(*dmax).a[idim])<1e-5*dr){
			for(unsigned int j=0; j<nvert; j++){
				if(j==i) continue;
				if(sqrt(pow(pos[i].a[(idim+1)%3]-pos[j].a[(idim+1)%3],(float)2.)+ \
				        pow(pos[i].a[(idim+2)%3]-pos[j].a[(idim+2)%3],(float)2.)+ \
								pow(pos[j].a[idim      ]- (*dmin).a[idim]      ,(float)2.) ) < 1e-4*dr){
					newlink[i] = j;
					//"delete" vertex
					for(int k=0; k<3; k++)
						pos[i].a[k] = -1e10;
					break;
				}
				if(j==nvert-1){
					// cout << " [FAILED]" << endl;
					return; //NO_PER_VERT;
				}
			}
		}
		i += blockDim.x*gridDim.x;
	}

//	gpu_sync(sync_i, sync_o);
	if(i_c==0){
		lock.lock();
		lock.unlock();
	}
	__syncthreads();

	//relink
	i = blockIdx.x*blockDim.x+threadIdx.x;
	while(i<nbe){
    for(int j=0; j<3; j++){
		  if(newlink[ep[i].a[j]] != -1)
        ep[i].a[j] = newlink[ep[i].a[j]];
    }
		i += blockDim.x*gridDim.x;
	}

	return;
}

#ifndef bdebug
__global__ void calc_vert_volume (uf4 *pos, uf4 *norm, ui4 *ep, float *vol, int *trisize, uf4 *dmin, uf4 *dmax, int *sync_i, int *sync_o, int nvert, int nbe, float dr, float eps, bool *per, Lock lock)
#else
__global__ void calc_vert_volume (uf4 *pos, uf4 *norm, ui4 *ep, float *vol, int *trisize, uf4 *dmin, uf4 *dmax, int *sync_i, int *sync_o, int nvert, int nbe, float dr, float eps, bool *per, Lock lock, uf4 *debug, float* debugp)
#endif
{
	//get neighbouring vertices
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int i_c = threadIdx.x;
	while(i<nvert){
		trisize[i] = 0;
		i += blockDim.x*gridDim.x;
	}

//	gpu_sync(sync_i, sync_o);
	if(i_c==0){
		lock.lock();
		lock.unlock();
	}
	__syncthreads();

	i = blockIdx.x*blockDim.x+threadIdx.x;
	while(i<nbe){
		for(unsigned int j=0; j<3; j++){
			atomicAdd(&trisize[ep[i].a[j]],1);
		}
		i += blockDim.x*gridDim.x;
	}

//	gpu_sync(sync_i, sync_o);
	if(i_c==0){
		lock.lock();
		lock.unlock();
	}
	__syncthreads();

	//sort neighbouring vertices
	//calculate volume (geometry factor)
	unsigned int gsize = gres*2+1; //makes sure that grid is large enough
	float gdr = dr/(float)gres;
	float vgrid;
	float cvec[trimax][12][3];
	int tri[trimax][3];
	float avnorm[3];
	bool first[trimax];
	float vnorm;
	bool closed;
	int iduma[3];
	float sp;

	i = blockIdx.x*blockDim.x+threadIdx.x;
	while(i<nvert){

		//vertex has been deleted
		if(pos[i].a[0] < -1e9){
			i += blockDim.x*gridDim.x;
			continue;
		}

		//initialize variables
		closed = true;
		vol[i] = 0.;
		unsigned int tris = trisize[i];
    if(tris > trimax)
      return; //exception needs to be thrown
		for(unsigned int j=0; j<tris; j++)
			first[j] = true;
		for(unsigned int j=0; j<3; j++)
      avnorm[j] = 0.;

		//find connected faces
		unsigned int itris = 0;
		for(unsigned int j=0; j<nbe; j++){
			for(unsigned int k=0; k<3; k++){
				if(ep[j].a[k] == i){
					tri[itris][0] = ep[j].a[(k+1)%3];
					tri[itris][1] = ep[j].a[(k+2)%3];
					tri[itris][2] = j;
					itris++;
				}
			}
		}

		//try to put neighbouring faces next to each other
		for(unsigned int j=0; j<tris; j++){
			for(unsigned int k=j+1; k<tris; k++){
				if(tri[j][1] == tri[k][0]){
					if(k!=j+1){
						for(int l=0; l<3; l++){
							iduma[l] = tri[j+1][l];
							tri[j+1][l] = tri[k][l];
							tri[k][l] = iduma[l];
						}
					}
					break;
				}
				if(tri[j][1] == tri[k][1]){
					iduma[0] = tri[k][1];
					iduma[1] = tri[k][0];
					iduma[2] = tri[k][2];
					for(int l=0; l<3; l++){
						tri[k][l] = tri[j+1][l];
						tri[j+1][l] = iduma[l];
					}
					break;
				}
				if(k==tris-1) closed = false;
			}
		}
		if(tri[0][0] != tri[tris-1][1]){
			closed = false;
		}

		//start big loop over all numerical integration points
		for(unsigned int k=0; k<gsize; k++){
		for(unsigned int l=0; l<gsize; l++){
		for(unsigned int m=0; m<gsize; m++){

			float gp[3]; //gridpoint in coordinates relative to vertex
			gp[0] = (((float)k-(float)(gsize-1)/2))*gdr;
			gp[1] = (((float)l-(float)(gsize-1)/2))*gdr;
			gp[2] = (((float)m-(float)(gsize-1)/2))*gdr;
			vgrid = 0.;

#ifdef bdebug
			if(i==bdebug){
			for(int j=0; j<3; j++) debug[k+l*gsize+m*gsize*gsize].a[j] = gp[j] + pos[i].a[j];
			debug[k+l*gsize+m*gsize*gsize].a[3] = -1.;
			for(int j=0; j<100; j++) debugp[j] = 0.;
			}
			if(i==bdebug) debugp[0] = tris;
#endif

			//create cubes
			for(unsigned int j=0; j<tris; j++){
				if(k+l+m==0){
					//setting up cube directions
					for(unsigned int n=0; n<3; n++) cvec[j][2][n] = norm[tri[j][2]].a[n]; //normal of boundary element
					vnorm = 0.;
					for(unsigned int n=0; n<3; n++){
						cvec[j][0][n] = pos[tri[j][0]].a[n]-pos[i].a[n]; //edge 1
						if(per[n]&&fabs(cvec[j][0][n])>2*dr)	cvec[j][0][n] += sgn(cvec[j][0][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
						vnorm += pow(cvec[j][0][n],2);
					}
					vnorm = sqrt(vnorm);
					for(unsigned int n=0; n<3; n++) cvec[j][0][n] /= vnorm; 
					for(unsigned int n=0; n<3; n++)	cvec[j][1][n] = cvec[j][0][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][0][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge1
					vnorm = 0.;
					for(unsigned int n=0; n<3; n++){
						cvec[j][3][n] = pos[tri[j][1]].a[n]-pos[i].a[n]; //edge 2
						if(per[n]&&fabs(cvec[j][3][n])>2*dr)	cvec[j][3][n] += sgn(cvec[j][3][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
						vnorm += pow(cvec[j][3][n],2);
						avnorm[n] -= norm[tri[j][2]].a[n];
					}
					vnorm = sqrt(vnorm);
					for(unsigned int n=0; n<3; n++) cvec[j][3][n] /= vnorm; 
					for(unsigned int n=0; n<3; n++)	cvec[j][4][n] = cvec[j][3][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][3][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge2
				}
				//filling vgrid
				bool incube[5] = {false, false, false, false, false};
				for(unsigned int n=0; n<5; n++){
					sp = 0.;
					for(unsigned int o=0; o<3; o++) sp += gp[o]*cvec[j][n][o];
					if(fabs(sp)<=dr/2.+eps) incube[n] = true;
				}
				if((incube[0] && incube[1] && incube[2]) || (incube[2] && incube[3] && incube[4])){
					vgrid = 1.;
#ifdef bdebug
			if(i==bdebug) debug[k+l*gsize+m*gsize*gsize].a[3] = 1.;
#endif
					if(k+l+m!=0) break; //makes sure that in the first grid point we loop over all triangles j s.t. values are initialized correctly.
				}
			}
			//end create cubes

			//remove points based on planes (voronoi diagram & walls)
			float tvec[3][3];
			for(unsigned int j=0; j<tris; j++){
				if(vgrid<eps) break; //gridpoint already empty
				if(first[j]){
					first[j] = false;
					//set up plane normals and points
					for(unsigned int n=0; n<3; n++){
						cvec[j][5][n] = pos[tri[j][0]].a[n]-pos[i].a[n]; //normal of plane voronoi
						if(per[n]&&fabs(cvec[j][5][n])>2*dr)	cvec[j][5][n] += sgn(cvec[j][5][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
						cvec[j][6][n] = pos[i].a[n]+cvec[j][5][n]/2.; //position of plane voronoi
						tvec[0][n] = cvec[j][5][n]; // edge 1
						tvec[1][n] = pos[tri[j][1]].a[n]-pos[i].a[n]; // edge 2
						if(per[n]&&fabs(tvec[1][n])>2*dr)	tvec[1][n] += sgn(tvec[1][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
						if(!closed){
							cvec[j][7][n] = tvec[1][n]; //normal of plane voronoi 2
							cvec[j][8][n] = pos[i].a[n]+cvec[j][7][n]/2.; //position of plane voronoi 2
						}
						tvec[2][n] = avnorm[n]; // negative average normal
					}
					for(unsigned int n=0; n<3; n++){
						for(unsigned int k=0; k<3; k++){
							cvec[j][k+9][n] = tvec[k][(n+1)%3]*tvec[(k+1)%3][(n+2)%3]-tvec[k][(n+2)%3]*tvec[(k+1)%3][(n+1)%3]; //normals of tetrahedron planes
						}
					}
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += norm[tri[j][2]].a[n]*cvec[j][9][n]; //test whether normals point inward tetrahedron, if no flip normals
					if(sp > 0.){
						for(unsigned int k=0; k<3; k++){
							for(unsigned int n=0; n<3; n++)	cvec[j][k+9][n] *= -1.;
						}
					}
				}

			  //remove unwanted points and sum up for volume
				//voronoi plane
				for(unsigned int n=0; n<3; n++) tvec[0][n] = gp[n] + pos[i].a[n] - cvec[j][6][n];
				sp = 0.;
				for(unsigned int n=0; n<3; n++) sp += tvec[0][n]*cvec[j][5][n];
				if(sp>0.+eps){
					vgrid = 0.;
#ifdef bdebug
			if(i==bdebug) debug[k+l*gsize+m*gsize*gsize].a[3] = 0.;
#endif
					break;
				}
				else if(fabs(sp) < eps){
					vgrid /= 2.;
				}
				//voronoi plane 2
				if(!closed){
					for(unsigned int n=0; n<3; n++) tvec[0][n] = gp[n] + pos[i].a[n] - cvec[j][8][n];
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += tvec[0][n]*cvec[j][7][n];
					if(sp>0.+eps){
						vgrid = 0.;
						break;
					}
					else if(fabs(sp) < eps){
						vgrid /= 2.;
					}
				}
				//walls
				bool half = false;
				for(unsigned int o=0; o<3; o++){
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += gp[n]*cvec[j][9+o][n];
					if(sp<0.-eps) break;
					if(fabs(sp)<eps && o==0) half=true;
					if(o==2 && !half){
						vgrid = 0.;
#ifdef bdebug
			if(i==bdebug) debug[k+l*gsize+m*gsize*gsize].a[3] = 0.;
#endif
						break;
					}
					else if(o==2 && half){
						vgrid /= 2.;
					}
				}
				if(vgrid < eps) break;

				//volume sum
				if(j==tris-1)	vol[i] += vgrid;
			}

		}
		}
		}
		//end looping through all gridpoints

		//calculate volume
		vol[i] *= pow(dr/(float)gres,3);

		i += blockDim.x*gridDim.x;
	}
}

__global__ void fill_fluid (uf4 *fpos, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float eps, float dr, int *nfib, int fmax, Lock lock)
{
	//this can be a bit more complex in order to fill complex geometries
	__shared__ int nfib_cache[threadsPerBlock];
	int idim = (floor((ymax+eps-ymin)/dr)+1)*(floor((xmax+eps-xmin)/dr)+1);
	int jdim =  floor((xmax+eps-xmin)/dr)+1;
	int i, j, k, tmp, nfib_tmp;
	int tid = threadIdx.x;

	nfib_tmp = 0;
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	while(id<fmax){
		k = id/idim;
		tmp = id%idim;
		j = tmp/jdim;
		i = tmp%jdim;
		fpos[id].a[0] = xmin + (float)i*dr;
		fpos[id].a[1] = ymin + (float)j*dr;
		fpos[id].a[2] = zmin + (float)k*dr;
		nfib_tmp++;
		//if position should not be filled use a[0] = -1e10 and do not increment nfib_tmp
		id += blockDim.x*gridDim.x;
	}
	nfib_cache[tid] = nfib_tmp;

	__syncthreads();

	j = blockDim.x/2;
	while (j!=0){
		if(tid < j)
			nfib_cache[tid] += nfib_cache[tid+j];
		__syncthreads();
		j /= 2;
	}

	if(tid == 0){
		lock.lock();
		*nfib += nfib_cache[0];
		lock.unlock();
	}
}

//Implemented according to: Inter-Block GPU Communication via Fast Barrier Synchronization, Shucai Xiao and Wu-chun Feng, Department of Computer Science, Virginia Tech, 2009
//The original implementation doesn't work. For now lock is used.
__device__ void gpu_sync (int *sync_i, int *sync_o)
{
	int tid_in_block = threadIdx.x;
	int bid = blockIdx.x;
	int nblock = gridDim.x;

	//sync thread 0 in all blocks
	if(tid_in_block == 0){
		sync_i[bid] = 1;
		sync_o[bid] = 0;
	}
	
	if(bid == 0){
		int i = tid_in_block;
		while (i<nblock) {
			while(sync_i[i] != 1)
				;
			i += blockDim.x;
		}
		__syncthreads();

		i = tid_in_block;
		while (i<nblock) {
			sync_o[i] = 1;
			sync_i[i] = 0;
			i += blockDim.x;
		}
	}

//this last part causes an infinite loop, why?
	//sync block
	if(tid_in_block == 0 && bid == 1){
		while (sync_o[bid] != 1)
      			;
	}

	__syncthreads();

}

#endif
