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

__global__ void find_links(uf4 *pos, int nvert, uf4 *dmax, uf4 *dmin, float dr, int *newlink, int idim)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
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
}

//__device__ volatile int lock_per_mutex[2]={0,0};
__global__ void periodicity_links (uf4 *pos, ui4 *ep, int nvert, int nbe, uf4 *dmax, uf4 *dmin, float dr, int *newlink, int idim)
{
	//find corresponding vertices
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

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

__global__ void calc_trisize(ui4 *ep, int *trisize, int nbe)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	while(i<nbe){
		for(unsigned int j=0; j<3; j++){
			atomicAdd(&trisize[ep[i].a[j]],1);
		}
		i += blockDim.x*gridDim.x;
	}
}

//__device__ volatile int lock_mutex[2];
#ifndef bdebug
__global__ void calc_vert_volume (uf4 *pos, uf4 *norm, ui4 *ep, float *vol, int *trisize, uf4 *dmin, uf4 *dmax, int nvert, int nbe, float dr, float eps, bool *per)
#else
__global__ void calc_vert_volume (uf4 *pos, uf4 *norm, ui4 *ep, float *vol, int *trisize, uf4 *dmin, uf4 *dmax, int nvert, int nbe, float dr, float eps, bool *per, uf4 *debug, float* debugp)
#endif
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	//sort neighbouring vertices
	//calculate volume (geometry factor)
	unsigned int gsize = gres*2+1; //makes sure that grid is large enough
	float gdr = dr/(float)gres;
	float vgrid;
	float cvec[trimax][12][3];
	int tri[trimax][3];
	float avnorm[3];
	bool first[trimax];
	uf4 edgen[trimax];
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
		for(unsigned int j=0; j<tris; j++){
			first[j] = true;
			for(unsigned int k=0; k<3; k++)
				edgen[j].a[k] = 0.;
		}
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
		
		// calculate average normal at edge
		itris = 0;
		for(unsigned int j=0; j<nbe; j++){
			for(unsigned int k=0; k<tris; k++){
				if(edgen[k].a[3]==2)
					continue;
				int vfound = 0;
				for(unsigned int l=0; l<3; l++){
					if(ep[j].a[l] == tri[k][0] || ep[j].a[l] == tri[k][1])
						vfound++;
				}
				if(vfound==2){
					for(unsigned int l=0; l<3; l++)
						edgen[k].a[l] += norm[j].a[l];
					edgen[k].a[3]++;
				}
				if(edgen[k].a[3]==2){ //cross product to determine normal of wall
					int tmpvec[3], edge[3];
					for(unsigned int n=0; n<3; n++) edge[n] = pos[tri[k][0]] - pos[tri[k][1]];
					for(unsigned int n=0; n<3; n++)	tmpvec[n] = edgen[k].a[(n+1)%3]*edge[(n+2)%3]-edgen[k].a[(n+2)%3]*edge[(n+1)%3];
					for(unsigned int n=0; n<3; n++) edgen[k].a[n] = tmpvec[n];
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

#ifdef bdebug
			if(i==bdebug){
				for(int j=0; j<100; j++) debugp[j] = 0.;
				debugp[0] = tris;
			}
#endif

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
			}
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
					//edge normal to point in right direction
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += edgen[j].a[n]*cvec[j][5][n]; //sp of edge plane normal and vector pointing from vertex to plane point
					if(sp < 0.){
						for(unsigned int n=0; n<3; n++) edgen[j].a[n] *= -1.; //flip
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
				//edges
				sp = 0.;
				for(unsigned int n=0; n<3; n++){
					tvec[0][n] = gp[n] - cvec[j][5][n];
					sp += tvec[0][n]*edgen[j].a[n];
				}
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

__global__ void calc_ngridp (uf4 *pos, unsigned int *igrid, uf4 *dmin, uf4 *dmax, bool *per, int *ngridp, int maxgridp, float dr, float eps, int nvert, int nbe, float krad, Lock lock, int igrids){
	const unsigned int uibs = 8*sizeof(unsigned int);
	unsigned int byte[uibs];
	for(int i=0; i<uibs; i++)
		byte[i] = 1<<i;
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int i_c = threadIdx.x;
	int idim = (floor(((*dmax).a[1]+eps-(*dmin).a[1])/dr)+1)*(floor(((*dmax).a[0]+eps-(*dmin).a[0])/dr)+1);
	int jdim =  floor(((*dmax).a[0]+eps-(*dmin).a[0])/dr)+1;
	__shared__ int ngridpl[threadsPerBlock];
	ngridpl[i_c] = 0;

	while(id<maxgridp){
		int ipos[3];
		ipos[2] = id/idim;
		int tmp = id%idim;
		ipos[1] = tmp/jdim;
		ipos[0] = tmp%jdim;
		float gpos[3], rvec[3];
		for(int i=0; i<3; i++) gpos[i] = (*dmin).a[i] + ((float)ipos[i])*dr;
		for(int i=0; i<nvert+nbe; i++){
			bool bbreak = false;
			for(int j=0; j<3; j++){
				rvec[j] = gpos[j] - pos[i].a[j];
				//this introduces a min size for the domain, check it
				if(per[j]&&fabs(rvec[j])>2*(krad+dr))	rvec[j] += sgn(rvec[j])*(-(*dmax).a[j]+(*dmin).a[j]); //periodicity
				if(fabs(rvec[j]) > krad+dr+eps){
					bbreak = true;
					break;
				}
			}
			if(bbreak) continue;
			if(sqrt(sqr(rvec[0])+sqr(rvec[1])+sqr(rvec[2])) <= krad+dr+eps){
				ngridpl[i_c] += 1;
				int ida = id/uibs;
				int idi = id%uibs;
				unsigned int tbyte = byte[idi];
				atomicOr(&igrid[ida],tbyte);
				break;
			}
		}
		id += blockDim.x*gridDim.x;
	}

	__syncthreads();
	int j = blockDim.x/2;
	while (j!=0){
		if(i_c < j){
			ngridpl[i_c] += ngridpl[i_c+j];
		}
		__syncthreads();
		j /= 2;
	}

	if(i_c == 0){
		lock.lock();
		*ngridp += ngridpl[0];
		lock.unlock();
	} 
}

__device__ float rand(float seed){
	const unsigned long m = 1UL<<32; //2^32
	const unsigned long a = 1664525UL;
	const unsigned long c = 1013904223UL;
	unsigned long xn = (unsigned long) (seed*float(m));
	unsigned long xnp1 = (a*xn+c)%m;
	return (float)((float)xnp1/(float)m);
}

__device__ inline float wendland_kernel(float q, float h){
	const float alpha = 21./16./3.1415926;
	return alpha/(h*sqr(h))*sqr(sqr((1.-q/2.)))*(1.+2.*q);
}

__device__ int calc_ggam(uf4 tpos, uf4 *pos, ui4 *ep, float *surf, uf4 *norm, uf4 *gpos, float *ggam, int *iggam, uf4 *dmin, uf4 *dmax, bool *per, int ngridp, float dr, float hdr, int iker, float eps, int nvert, int nbe, float krad, float iseed, bool ongpoint, int id){
	//find neighbouring boundary elements
	int nlink = 0;
	int link[maxlink];
	float h = hdr*dr;
	for(int i=0; i<nvert+nbe; i++){
		float rvecn = 0.;
		for(int j=0; j<3; j++) rvecn += sqr(tpos.a[j]-pos[i].a[j]);
		if(sqrt(rvecn) <= krad+eps){
			if(i>=nvert){
				bool found = false;
				for(int j=0; j<nlink; j++){
					if(link[j] == i-nvert){
						found = true;
						break;
					}
				}
				if(!found){
					link[nlink] = i-nvert;
					nlink++;
					if(nlink>maxlink){
						//deb[8] = id;
						return MAXLINK_TOO_SMALL;
					}
				}
			}
			else{
				for(int j=0; j<nbe; j++){
					for(int k=0; k<3; k++){
						if(ep[j].a[k] == i){
							bool found = false;
							for(int l=0; l<nlink; l++){
								if(link[l] == j){
									found = true;
									break;
								}
							}
							if(!found){
								link[nlink] = j;
								nlink++;
								if(nlink>maxlink){
									//deb[8] = id;
									return MAXLINK_TOO_SMALL;
								}
							}
							break;
						}
					}
				}
			}
		}
	}
	//calculate ggam_{pb}
	for(int i=0; i<nlink; i++){
		int ib = link[i];
		float nggam = 0.;
		uf4 verts[3];
		for(int j=0; j<3; j++){
			for(int k=0; k<3; k++)
				verts[j].a[k] = pos[ep[ib].a[j]].a[k];
		}
		for(int j=0; j<3; j++){
			if(ongpoint) ggam[id*maxlink*3+i*3+j]  = 0.;
		}
		for(int j=0; j<ipoints; j++){
			int it;
			if(ipoints < 1)
				it = 0;
			else if(ipoints < 8)
				it = 3;
			else
				it = 6;
			float wei = intri[j].a[0];
			float bpos[3];
			for(int k=0; k<3; k++)
				bpos[k] = intri[j].a[k+1];
			int per[3];
			for(int l=0; l<it; l++){
				uf4 tp;
				per[0] = l%3;
				per[1] = (l+1+l/3)%3;
				per[2] = (l+2-l/3)%3;
				for(int k=0; k<3; k++){
					tp.a[k] = 0.
					for(int m=0; m<3; m++)
						tp.a[k] += bpos[j].a[per[m]]*verts[m].a[k];
				}
				float q = 0.;
				for(int k=0; k<3; k++)
					q += sqr(tp.a[k]-tpos.a[k]);
				q = sqrt(q)/h;
				switch(iker){
					case 1:
					default:
						nggam += wei*wendland_kernel(q,h);
				}
			}
		}
		for(int j=0; j<3; j++){
			if(ongpoint){
				ggam[id*maxlink*3+i*3+j]  += nggam * surf[ib] * norm[ib].a[j];
			}
			else{
				ggam[j] += nggam * surf[ib] * norm[ib].a[j];
			}
		}
		if(ongpoint)
			iggam[id*maxlink+i] = ib;
	}
	return nlink;
}

__global__ void init_gpoints (uf4 *pos, ui4 *ep, float *surf, uf4 *norm, uf4 *gpos, float *gam, float *ggam, int *iggam, uf4 *dmin, uf4 *dmax, bool *per, int ngridp, float dr, float hdr, int iker, float eps, int nvert, int nbe, float krad, float seed, int *nrggam, Lock lock, float *deb, int *ilock){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int i_c = threadIdx.x;
	deb[id] = 0.;
	ilock[id] = 0.;
	__shared__ int nrggaml[threadsPerBlock];
	nrggaml[i_c] = 0;
	int dim[3];
	for(int i=0; i<3; i++)
		dim[i] = (floor(((*dmax).a[i]+eps-(*dmin).a[i])/dr)+1);
	//initializing random number generator
	float iseed = (float)id/((float)(blockDim.x*gridDim.x))+seed;
	if(iseed > 1.)
		iseed -= 1.;
	iseed = rand(iseed);
	for(int i=0; i<(int)(iseed*20.); i++)
		iseed = rand(iseed);
	//calculate position and neighbours, initialize gam for those who don't have neighbours
	//find boundary elements and calculate ggam_{pb}
	while(id < ngridp){
		//calculate position
		int ipos[3];
		uf4 tpos;
		ipos[2] = (int)(gpos[id].a[3])/(dim[0]*dim[1]);
		int tmp = (int)(gpos[id].a[3])%(dim[0]*dim[1]);
		ipos[1] = tmp/dim[0];
		ipos[0] = tmp%dim[0];
		for(int i=0; i<3; i++){
			gpos[id].a[i] = (*dmin).a[i] + ((float)ipos[i])*dr;
			tpos.a[i] = gpos[id].a[i];
		}

		//calculate ggam at tpos
		int nlink = calc_ggam(tpos, pos, ep, surf, norm, gpos, ggam, iggam, dmin, dmax, per, ngridp, dr, hdr, iker, eps, nvert, nbe, krad, iseed, true, id);

		for(int i=nlink; i<maxlink; i++){
			ggam[id*maxlink*3+i*3] = -1e10;
			iggam[id*maxlink+i] = -1;
		}
		nrggaml[i_c] += nlink;
		id+=blockDim.x*gridDim.x;
	} 
	//calculate gamma
	//find the ones with gamma = 1
	id = blockIdx.x*blockDim.x+threadIdx.x;
	while(id<ngridp){
		bool found = false;
		float mindist = 1e10;
		int imin = -1;
		for(int i=0; i<nvert+nbe; i++){
			float dist = 0.;
			for(int j=0; j<3; j++)
				dist += sqr(pos[i].a[j]-gpos[id].a[j]);
			if(dist < mindist && i >= nvert){
				mindist = dist;
				imin = i-nvert;
			}
			if(sqrt(dist) < krad+eps){
				found = true;
				break;
			}
		}
		if(!found){
			//ensure that grid point is inside the fluid
			float sp = 0.;
			for(int i=0; i<3; i++)
				sp += (gpos[id].a[i]-pos[imin+nvert].a[i])*norm[imin].a[i];
			if(sp>0.){
				gam[id] = 1.;
			}
			else{
				gam[id] = 0.;
			}
			atomicExch(ilock+id,0);
		}
		else{
			gam[id] = -1.;
			atomicExch(ilock+id,1);
		}
		id+=blockDim.x*gridDim.x;
	}

	//for some reason I have to use the atomicAdd function here. why?
	atomicAdd(nrggam,nrggaml[i_c]);
	/*__syncthreads();
	int j = blockDim.x/2;
	while (j!=0){
		if(i_c < j)
			nrggaml[i_c] += nrggaml[i_c+j];
		__syncthreads();
		j /= 2;
	}

	if(i_c == 0){
		lock.lock();
		*nrggam += nrggaml[0];
		lock.unlock();
	}*/
}

__global__ void lobato_gpoints (uf4 *pos, ui4 *ep, float *surf, uf4 *norm, uf4 *gpos, float *gam, float *ggam, int *iggam, uf4 *dmin, uf4 *dmax, bool *per, int ngridp, float dr, float hdr, int iker, float eps, int nvert, int nbe, float krad, float seed, int *nrggam, Lock lock, float *deb, int *ilock)
{
	//calculate the others using Lobato
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int dim[3];
	for(int i=0; i<3; i++)
		dim[i] = (floor(((*dmax).a[i]+eps-(*dmin).a[i])/dr)+1);

	//initializing random number generator
	float iseed = (float)id/((float)(blockDim.x*gridDim.x))+seed;
	if(iseed > 1.)
		iseed -= 1.;
	iseed = rand(iseed);
	for(int i=0; i<(int)(iseed*20.); i++)
		iseed = rand(iseed);

	while(id<ngridp){
		int ij=0;
		//find neighbouring grid points, don't look for diagonal ones, shouldn't be necessary
		//there is the possibility of a very unlikely case that the higher id's need to be calculated first. so blocking here is not the ideal option but it should work
		int neibs[6], idneibs[6];
		int ipos[3];
		ipos[2] = (int)(gpos[id].a[3])/(dim[0]*dim[1]);
		int tmp = (int)(gpos[id].a[3])%(dim[0]*dim[1]);
		ipos[1] = tmp/dim[0];
		ipos[0] = tmp%dim[0];
		//get id of neigbouring grid points
		for(int i=0; i<6; i++){
			int di[3];
			for(int j=0; j<3; j++)
				di[j] = ((i%2)*2-1)*((i/2==j)?1:0);
			if(ipos[i/2]==(dim[i/2]-1)*(i%2) && !per[i/2]){
				idneibs[i] = -1;
			}
			else{
				idneibs[i] = (ipos[0]+di[0])%dim[0] + ((ipos[1]+di[1])%dim[1])*dim[0] + ((ipos[2]+di[2])%dim[2])*dim[0]*dim[1];
			}
			neibs[i] = -1;
		}
		int ii = 0;
		for(int i=0; i<ngridp; i++){
			if(i==id)
				continue;
			for(int j=0; j<6; j++){
				if(idneibs[j] == (int)(gpos[i].a[3])){
					neibs[j] = i;
					ii++;
					break;
				}
			}
			if(ii==6)
				break;
		}
		bool gamcalc = (gam[id] > -0.5);
		while(!gamcalc){
			float W[7],P[6];
			W[0] = 0.0476190476;
			W[1] = 0.2768260473;
			W[2] = 0.4317453812;
			W[3] = 0.4876190476;
			W[4] = 0.4317453812;
			W[5] = 0.2768260473;
			W[6] = 0.0476190476;
			P[1] =-0.8302238962;
			P[2] =-0.4688487934;
			P[3] = 0.0000000000;
			P[4] = 0.4688487934;
			P[5] = 0.8302238962;
			for(int i=0; i<6; i++){
				if(neibs[i] < 0)
					continue;
				int locked = 1;
				locked = atomicAnd(ilock + (neibs[i]), locked);
				if(locked == 0 && gam[neibs[i]] > 1e-9 ){
					atomicExch(ilock+(id),1);
					uf4 rvec, mid;
					float tggam[3];
					int idn = neibs[i];
					for(int j=0; j<3; j++){
						rvec.a[j]  = (gpos[id].a[j]-gpos[idn].a[j])/2.;
						mid.a[j]   = (gpos[id].a[j]+gpos[idn].a[j])/2.;
						tggam[j] = 0.;
					}
					//end points
					for(int j=0; j<maxlink; j++){
						int end = 0;
						if(ggam[id*maxlink*3+j*3] > -1e9){
							for(int k=0; k<3; k++)
								tggam[k] += ggam[id*maxlink*3+j*3+k];
						}
						else{
							end++;
						}
						if(ggam[idn*maxlink*3+j*3] > -1e9){
							for(int k=0; k<3; k++)
								tggam[k] += ggam[idn*maxlink*3+j*3+k];
						}
						else{
							end++;
						}
						if(end==2)
							break;
					}
					float tgam;
					tgam = gam[idn];
					for(int j=0; j<3; j++)
						tgam += W[0]*(tggam[j]*rvec.a[j]);
					//interior points
					for(int j=1; j<=5; j++){
						uf4 tp;
						for(int k=0; k<3; k++){
							tp.a[k] = mid.a[k]+P[j]*rvec.a[k];
							tggam[k] = 0.;
						}

						//calc ggam at tp
						int nlink = calc_ggam(tp, pos, ep, surf, norm, gpos, tggam, iggam, dmin, dmax, per, ngridp, dr, hdr, iker, eps, nvert, nbe, krad, iseed, false, 0);

						for(int k=0; k<3; k++)
							tgam += W[j]*(tggam[k]*rvec.a[k]);
					}
					//end of gam calculation
					tgam = fmin(1.f,tgam);
					/*if(id==6733){
						deb[0] = idn;
						deb[1] = gam[idn];
						deb[2] = gam[id];
						deb[3] = (float) ilock[idn];
						deb[4] = (float) ilock[id];
						deb[5] = tgam;
						for(int k=0; k<3; k++){
							deb[6+k] = tggam[k];
						}
					}*/
					if(tgam > 1e-8 || ij > 200){
						if(ij > 200)
							tgam = fmax(1e-8f, tgam);
						deb[id] = ij;
						gam[id] = tgam;
						gamcalc = true;
						if(gam[id]>0.)
							atomicExch((ilock+id),0);
						break;
					}
					ij++; // not sure why ij is necessary but it certainly works
				}
			}
		}
		id+=blockDim.x*gridDim.x;
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

#endif
