#ifndef CRIXUS_D_CU
#define CRIXUS_D_CU

#include <cuda.h>
#include "lock.cuh"
#include "crixus_d.cuh"
#include "return.h"
#include "crixus.h"
#include "vector_math.h"

// dot product (3D) which allows vector operations in arguments
#define dot(u,v)   ((u).a[0] * (v).a[0] + (u).a[1] * (v).a[1] + (u).a[2] * (v).a[2])

__global__ void set_bound_elem (uf4 *pos, uf4 *norm, float *surf, ui4 *ep, unsigned int nbe, float *xminp, float *xminn, float *nminp, float*nminn, Lock lock, int nvert)
{
  float ddum[3];
  uf4 ndir;
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
      // get normal spanned by v1-v0 and v2-v0
      ndir.a[j] = ((pos[ep[i].a[1]].a[(j+1)%3]-pos[ep[i].a[0]].a[(j+1)%3])*
                   (pos[ep[i].a[2]].a[(j+2)%3]-pos[ep[i].a[0]].a[(j+2)%3]))-
                  ((pos[ep[i].a[1]].a[(j+2)%3]-pos[ep[i].a[0]].a[(j+2)%3])*
                   (pos[ep[i].a[2]].a[(j+1)%3]-pos[ep[i].a[0]].a[(j+1)%3]));

    }
    // check if vertices are set in a clockwise fashion
    // if yes change vertex 1 and 2
    // this ensures that in GPUSPH (v_{10} x n_s) points out of the segment
    if(dot(norm[i],ndir) < 0.0f){
      int tmp = ep[i].a[1];
      ep[i].a[1] = ep[i].a[2];
      ep[i].a[2] = tmp;
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
    if(fabs(pos[i].a[idim]-(*dmax).a[idim])<1e-4*dr){
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
  float gdr = dr/2.0/(float)gres;
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
      for(unsigned int k=0; k<4; k++)
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
#ifdef bdebug
//        for(int j=0; j<tris; j++){
          if(i==bdebug){
          debugp[4+itris*4+0] = ep[j].a[0]+960;
          debugp[4+itris*4+1] = ep[j].a[1]+960;
          debugp[4+itris*4+2] = ep[j].a[2]+960;
          debugp[4+itris*4+3] = tri[itris][2]+2498;
        }
//        }
#endif
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

    // calculate average normal at edge
    itris = 0;
    for(unsigned int j=0; j<nbe; j++){
      // only use segments which are close enough
      if(sqlength3(perCorPos(pos[i]-pos[j+nvert],per,dmax,dmin)) < dr*dr*9){
        for(unsigned int k=0; k<tris; k++){
          if((int)(edgen[k].a[3]+eps)==2)
            continue;
          int vfound = 0;
          for(unsigned int l=0; l<3; l++){
            if(ep[j].a[l] == tri[k][0] || ep[j].a[l] == tri[k][1])
              vfound++;
          }
          if(vfound==2){
            for(unsigned int l=0; l<3; l++)
              edgen[k].a[l] += norm[j].a[l];
            edgen[k].a[3]+=1.;
          }
          if((int)(edgen[k].a[3]+eps)==2){ //cross product to determine normal of wall
            uf4 edge;
            edge = pos[tri[k][0]] - pos[tri[k][1]];
            for(unsigned int n=0; n<3; n++)
              if(per[n]&&fabs(edge.a[n])>2*dr)  edge.a[n] += sgn(edge.a[n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
            edgen[k] = cross(edgen[k], edge);
            itris++;
          }
        }
        if(itris == tris)
          break;
      }
    }

#ifdef bdebug
      if(i==bdebug){
//        for(int j=0; j<100; j++) debugp[j] = 0.;
        debugp[0] = tris;
        debugp[1] = pos[i].a[0];
        debugp[2] = pos[i].a[1];
        debugp[3] = pos[i].a[2];
      }
#endif

    bool cubesInitialized = false;

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
        if(!cubesInitialized){
          //setting up cube directions
          for(unsigned int n=0; n<3; n++) cvec[j][2][n] = norm[tri[j][2]].a[n]; //normal of boundary element
          vnorm = 0.;
          for(unsigned int n=0; n<3; n++){
            cvec[j][0][n] = pos[tri[j][0]].a[n]-pos[i].a[n]; //edge 1
            if(per[n]&&fabs(cvec[j][0][n])>2*dr)  cvec[j][0][n] += sgn(cvec[j][0][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
            vnorm += pow(cvec[j][0][n],2);
          }
          vnorm = sqrt(vnorm);
          for(unsigned int n=0; n<3; n++) cvec[j][0][n] /= vnorm;
          for(unsigned int n=0; n<3; n++)  cvec[j][1][n] = cvec[j][0][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][0][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge1
          vnorm = 0.;
          for(unsigned int n=0; n<3; n++){
            cvec[j][3][n] = pos[tri[j][1]].a[n]-pos[i].a[n]; //edge 2
            if(per[n]&&fabs(cvec[j][3][n])>2*dr)  cvec[j][3][n] += sgn(cvec[j][3][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
            vnorm += pow(cvec[j][3][n],2);
            avnorm[n] -= norm[tri[j][2]].a[n];
          }
          vnorm = sqrt(vnorm);
          for(unsigned int n=0; n<3; n++) cvec[j][3][n] /= vnorm;
          for(unsigned int n=0; n<3; n++)  cvec[j][4][n] = cvec[j][3][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][3][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge2
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
          if(cubesInitialized) break; //makes sure that in the first grid point we loop over all triangles j s.t. values are initialized correctly.
        }
      }
      //end create cubes
      cubesInitialized = true;

      //remove points based on planes (voronoi diagram & walls)
      float tvec[3][3];
      for(unsigned int j=0; j<tris; j++){
        if(vgrid<eps) break; //gridpoint already empty
        if(first[j]){
          first[j] = false;
          //set up plane normals and points
          for(unsigned int n=0; n<3; n++){
            cvec[j][5][n] = pos[tri[j][0]].a[n]-pos[i].a[n]; //normal of plane voronoi
            if(per[n]&&fabs(cvec[j][5][n])>2*dr)  cvec[j][5][n] += sgn(cvec[j][5][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
            cvec[j][6][n] = pos[i].a[n]+cvec[j][5][n]/2.; //position of plane voronoi
            tvec[0][n] = cvec[j][5][n]; // edge 1
            tvec[1][n] = pos[tri[j][1]].a[n]-pos[i].a[n]; // edge 2
            if(per[n]&&fabs(tvec[1][n])>2*dr)  tvec[1][n] += sgn(tvec[1][n])*(-(*dmax).a[n]+(*dmin).a[n]); //periodicity
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
              for(unsigned int n=0; n<3; n++)  cvec[j][k+9][n] *= -1.;
            }
          }
          //edge normal to point in right direction
          sp = 0.;
          for(unsigned int n=0; n<3; n++) sp += edgen[j].a[n]*cvec[j][5][n]; //sp of edge plane normal and vector pointing from vertex to plane point
          if(sp < 0.){
            for(unsigned int n=0; n<3; n++) edgen[j].a[n] *= -1.; //flip
          }
#ifdef bdebug
//        for(int j=0; j<tris; j++){
          if(i==bdebug){
          debugp[4+j*4+0] = edgen[j].a[0];
          debugp[4+j*4+1] = edgen[j].a[1];
          debugp[4+j*4+2] = edgen[j].a[2];
          debugp[4+j*4+3] = tri[j][2]+2498;
          }
//        }
#endif
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
      if(i==bdebug) debug[k+l*gsize+m*gsize*gsize].a[3] = -0.5;
#endif
          break;
        }
        else if(fabs(sp) < eps){
          vgrid /= 2.;
        }
        if(vgrid < eps) break;

        //volume sum
        if(j==tris-1)  vol[i] += vgrid;
      }

    }
    }
    }
    //end looping through all gridpoints

    //calculate volume
    vol[i] *= pow(dr/2.0/(float)gres,3);

    i += blockDim.x*gridDim.x;
  }
}

__global__ void fill_fluid (unsigned int *fpos, unsigned int *nfi, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, uf4 *dmin, uf4 *dmax, float eps, float dr, Lock lock)
{
  // this function is responsible for filling a box with fluid
  __shared__ int nfi_cache[threadsPerBlock];
  int bitPerUint = 8*sizeof(unsigned int);
  //dim local
  int idim =  floor((xmax+eps-xmin)/dr)+1;
  int jdim =  floor((ymax+eps-ymin)/dr)+1;
  int kdim =  floor((zmax+eps-zmin)/dr)+1;
  //dim global
  int idimg = int(floor(((*dmax).a[0]-(*dmin).a[0]+eps)/dr+1));
  int jdimg = int(floor(((*dmax).a[1]-(*dmin).a[1]+eps)/dr+1));
  int kdimg = int(floor(((*dmax).a[2]-(*dmin).a[2]+eps)/dr+1));
  //min indices
  int imin = (int)round(((float)idimg-1.)*(xmin-(*dmin).a[0])/((*dmax).a[0]-(*dmin).a[0]));
  int jmin = (int)round(((float)jdimg-1.)*(ymin-(*dmin).a[1])/((*dmax).a[1]-(*dmin).a[1]));
  int kmin = (int)round(((float)kdimg-1.)*(zmin-(*dmin).a[2])/((*dmax).a[2]-(*dmin).a[2]));

  int arrayInd = blockIdx.x*blockDim.x+threadIdx.x;
  int i,j,k,tmp,nfi_tmp;
  nfi_tmp = 0;
  while(((int)ceil(arrayInd/((float)bitPerUint)))<idimg*jdimg*kdimg){
    // loop through all bits of the array entry with index arrayInd
    for(int ii=0, bitInd=arrayInd*bitPerUint; ii<bitPerUint; ii++, bitInd++){
      k = bitInd/(idimg*jdimg);
      tmp = bitInd%(idimg*jdimg);
      j = tmp/idimg;
      i = tmp%idimg;
      // are we in the big box?
      if(i<idimg && j<jdimg && k<kdimg){
        // are we in the small box that needs to be filled?
        if((     i>=imin &&      j>=jmin &&      k>=kmin) &&
           (i-imin< idim && j-jmin< jdim && k-kmin< kdim)){
          unsigned int bit = 1<<ii;
          fpos[arrayInd] = fpos[arrayInd] | bit;
          nfi_tmp++;
        }
      }
      else
        break;
    }
    arrayInd += blockDim.x*gridDim.x;
  }

  // now sum up all the filled bits
  int tid = threadIdx.x;
  nfi_cache[tid] = nfi_tmp;

  __syncthreads();

  j = blockDim.x/2;
  while (j!=0){
    if(tid < j)
      nfi_cache[tid] += nfi_cache[tid+j];
    __syncthreads();
    j /= 2;
  }

  if(tid == 0){
    lock.lock();
    *nfi += nfi_cache[0];
    lock.unlock();
  }
  return;
}

__global__ void fill_fluid_complex (unsigned int *fpos, unsigned int *nfi, uf4 *norm, ui4 *ep, uf4 *pos, int nbe, uf4 *dmin, uf4 *dmax, float eps, float dr, int sInd, Lock lock, bool bcoarse, int cnbe, float dr_wall, int iteration)
{
  // this function is responsible for filling a complex geometry with fluid
  const unsigned int bitPerUint = 8*sizeof(unsigned int);

  // place seed point
  int nfi_tmp = 0;
  if(iteration == 1 && threadIdx.x==0 && blockIdx.x==0)
  {
    int sIndex = sInd/bitPerUint;
    unsigned int sBit = 1<<(sInd%bitPerUint);
    if(!(fpos[sIndex] & sBit)){
      nfi_tmp++;
      fpos[sIndex] = fpos[sIndex] | sBit;
    }
  }

  __shared__ int nfi_cache[threadsPerBlock];
  //dim global
  ui4 dimg;
  dimg.a[0] = int(floor(((*dmax).a[0]-(*dmin).a[0]+eps)/dr+1));
  dimg.a[1] = int(floor(((*dmax).a[1]-(*dmin).a[1]+eps)/dr+1));
  dimg.a[2] = int(floor(((*dmax).a[2]-(*dmin).a[2]+eps)/dr+1));

  int arrayInd = blockIdx.x*blockDim.x+threadIdx.x;
  int i,j,k;

  // indices of seed point
  int is = -1;
  int js = -1;
  int ks = -1;
  int tmp;
  if(iteration==1){
    ks = sInd/(dimg.a[0]*dimg.a[1]);
    tmp = sInd%(dimg.a[0]*dimg.a[1]);
    js = tmp/dimg.a[0];
    is = tmp%dimg.a[0];
  }

  while(arrayInd<(int)ceil(((float)dimg.a[0]*dimg.a[1]*dimg.a[2])/((float)bitPerUint))){
    // loop through all bits of the array entry with index arrayInd
    for(int ii=0, bitInd=arrayInd*bitPerUint; ii<bitPerUint; ii++, bitInd++){
      unsigned int bit = 1<<ii;
      //check if position is already filled
      if(!(fpos[arrayInd] & bit)){
        k = bitInd/(dimg.a[0]*dimg.a[1]);
        tmp = bitInd%(dimg.a[0]*dimg.a[1]);
        j = tmp/dimg.a[0];
        i = tmp%dimg.a[0];
        // are we in the big box?
        if(i<dimg.a[0] && j<dimg.a[1] && k<dimg.a[2]){
          // in iteration 1 check direckt line of sight to seed point
          if(iteration==1){
            // note that in the following check we check all triangles, regardless of their distance
            bool collision = checkCollision(i,j,k,is,js,ks, norm, ep, pos, nbe, dr, dmin, dimg, eps, false, cnbe, dr_wall);
            if(!collision){
              fpos[arrayInd] = fpos[arrayInd] | bit;
              nfi_tmp++;
              // if we fill this particle, then for the next bit we can use the present as seed point
              // as this one is closer we don't have to loop over so many particles
              is = i;
              js = j;
              ks = k;
              continue;
            }
          }
          //look around
          for(int ij=0; ij<6; ij++){
            int ioff = 0, joff = 0, koff = 0;
            if(ij<=1) ioff = (ij==0) ? -1 : 1;
            if(ij>=4) koff = (ij==4) ? -1 : 1;
            if(ioff+koff==0) joff = (ij==2) ? -1 : 1;
            ioff += i;
            joff += j;
            koff += k;
            if(ioff<0 || ioff>=dimg.a[0] ||
               joff<0 || joff>=dimg.a[1] ||
               koff<0 || koff>=dimg.a[2]   )
              continue;
            int indOff = ioff + joff*dimg.a[0] + koff*dimg.a[0]*dimg.a[1];
            // check whether position is filled
            if(fpos[indOff/bitPerUint] & (1<<(indOff%bitPerUint))){
              // check for collision with triangle
              bool collision = checkCollision(i,j,k,ioff,joff,koff, norm, ep, pos, nbe, dr, dmin, dimg, eps, bcoarse, cnbe, dr_wall);
              if(!collision){
                fpos[arrayInd] = fpos[arrayInd] | bit;
                nfi_tmp++;
                break;
              }
            }
          }
        }
        else // no longer in big box
          break;
      }
    }
    arrayInd += blockDim.x*gridDim.x;
  }

  // now sum up all the filled bits
  int tid = threadIdx.x;
  nfi_cache[tid] = nfi_tmp;

  __syncthreads();

  j = blockDim.x/2;
  while (j!=0){
    if(tid < j)
      nfi_cache[tid] += nfi_cache[tid+j];
    __syncthreads();
    j /= 2;
  }

  if(tid == 0){
    lock.lock();
    *nfi += nfi_cache[0];
    lock.unlock();
  }

  return;
}

__device__ bool checkCollision(int si, int sj, int sk, int ei, int ej, int ek, uf4 *norm, ui4 *ep, uf4 *pos, int nbe, float dr, uf4* dmin, ui4 dimg, float eps, bool bcoarse, int cnbe, float dr_wall){
  // checks whether the line-segment determined by s. and e. intersects any available triangle and whether s is too close to any triangle
  bool collision = false;
  uf4 s, e, n, v[3], dir;
  s.a[0] = ((float)si)*dr + (*dmin).a[0]; // s is the position to be filled
  s.a[1] = ((float)sj)*dr + (*dmin).a[1];
  s.a[2] = ((float)sk)*dr + (*dmin).a[2];
  e.a[0] = ((float)ei)*dr + (*dmin).a[0]; // e is the already filled position
  e.a[1] = ((float)ej)*dr + (*dmin).a[1];
  e.a[2] = ((float)ek)*dr + (*dmin).a[2];
  for(int i=0; i<3; i++)
    dir.a[i] = e.a[i] - s.a[i];

  // the maximum distance for influence is equal to |(s-e)/2| + 3 * dr + dr_wall, the factor 3 is chosen a bit too large to make sure everything is ok
  float sqMaxInflDist = length3((s-e)/2.0f) + 3.0f*dr + dr_wall;
  sqMaxInflDist *= sqMaxInflDist;
  // loop through all triangles
  for(int i=0; i<nbe; i++){
    n = norm[i]; // normal of the triangle
    for(int j=0; j<3; j++)
      v[j] = pos[ep[i].a[j]];
    // if we don't have a coarse grid, the triangle needs to be close enough so that something can happen
    if(!bcoarse && i<nbe-cnbe){
      // dist = square distance from one vertex to the midpoint between s and e
      float dist = sqlength3((s+e)/2.0f - v[0]);
      if(dist > sqMaxInflDist)
        continue;
    }

    // check if we are too close to a boundary
    if(i<nbe-cnbe){
      uf4 segpos;
      uf4 normals[3]; // these are the normal vector of the edge towards the segment
      // segpos is the position of the segment
      segpos = (v[0] + v[1] + v[2])/3.0f;
      // first normals are the edge vectors
      normals[0] = v[1] - v[0];
      // .a[3] contains the length
      normals[0].a[3] = length3(normals[0]);
      normals[1] = v[2] - v[1];
      normals[1].a[3] = length3(normals[1]);
      normals[2] = v[0] - v[2];
      normals[2].a[3] = length3(normals[2]);

      // normalize the vectors
      normals[0] = normals[0] / normals[0].a[3];
      normals[1] = normals[1] / normals[1].a[3];
      normals[2] = normals[2] / normals[2].a[3];

      // .a[3] contains the dot product of the edge vector and segpos - starting vertex
      normals[0].a[3] = dot3(normals[0],segpos-v[0]);
      normals[1].a[3] = dot3(normals[1],segpos-v[1]);
      normals[2].a[3] = dot3(normals[2],segpos-v[2]);

      // from the vector (segpos - starting vertex) subtract the edge vector component to obtain the normal
      normals[0] = (segpos - v[0]) - normals[0]*normals[0].a[3];
      normals[1] = (segpos - v[1]) - normals[1]*normals[1].a[3];
      normals[2] = (segpos - v[2]) - normals[2]*normals[2].a[3];

      // projection of s onto the triangle plane
      uf4 projpos;
      // currently the vector from segment to s
      projpos = s - segpos;
      // dot product between the vector above and the normal
      projpos.a[3] = dot3(projpos,n);
      // projected postion of s on the triangle plane with respect to segpos
      projpos = projpos - n*projpos.a[3];

      // dot products to check in which region we are
      uf4 orientations;
      orientations.a[0] = dot(projpos + segpos - v[0],normals[0]);
      orientations.a[1] = dot(projpos + segpos - v[1],normals[1]);
      orientations.a[2] = dot(projpos + segpos - v[2],normals[2]);

      // projection inside the triangle
      float dist = 0.0;
      if(orientations.a[0] >= 0.0 && orientations.a[1] >= 0.0 && orientations.a[2] >= 0.0){
        // distance is equal to the distance between the projection and s
        dist = sqlength3(projpos - (s - segpos));
      }
      // two orientations are negative
      else if(orientations.a[1]*orientations.a[2]*orientations.a[3] > 0.0){
        // distance is equal to the distance between s and the closest vertex
        uf4 close;
        if(orientations.a[0] > 0.0)
          close = v[2];
        else if(orientations.a[1] > 0.0)
          close = v[0];
        else
          close = v[1];
        dist = sqlength3(s - close);
      }
      // only one orientations is negative (all three is not possible)
      else{
        // vertices associated with the edge closest to s
        uf4 x1;
        uf4 x2;
        if(orientations.a[0] < 0.0){
          x1 = v[0];
          x2 = v[1];
        }
        else if(orientations.a[1] < 0.0){
          x1 = v[1];
          x2 = v[2];
        }
        else{
          x1 = v[2];
          x2 = v[0];
        }
        // distance is equal to the distance between edge and s
        uf4 x21 = x2 - x1;
        uf4 sx1 = s - x1;
        float tdot = dot(x21,sx1);
        dist = sqlength3(sx1 - x21/sqlength3(x21)*tdot);
      }
      if(dist + eps < dr_wall*dr_wall)
        return true;
    }

    collision = checkTriangleCollision(s, dir, n, v, eps);

    if(collision)
      return true;
  }

  return collision;
}

// Copyright 2001, softSurfer (www.softsurfer.com)
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

__device__ bool checkTriangleCollision(uf4 s, uf4 dir, uf4 n, uf4 *vert, float eps){
    uf4    u, v;                              // triangle vectors
    uf4    w0, w;                        // ray vectors
    float  r, a, b;                           // params to calc ray-plane intersect

    // get triangle edge vectors and plane normal
    for(int j=0; j<3; j++){
      u.a[j] = vert[1].a[j] - vert[0].a[j];
      v.a[j] = vert[2].a[j] - vert[0].a[j];
      w0.a[j] = s.a[j] - vert[0].a[j];
    }
    a = -dot(n,w0);
    b = dot(n,dir);
    if (fabs(b) < eps) {                // ray is parallel to triangle plane
        if (fabs(a) < eps)              // ray lies in triangle plane
            return true;
        return false;                         // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < -eps || r > 1.0 + eps) // intersection point not on triangle plane
        return false;                         // => no intersect

    for(int j=0; j<3; j++)
      w.a[j] = s.a[j] + r*dir.a[j] - vert[0].a[j]; // vector from v0 to intersection point

    // is I inside T?
    float    uu, uv, vv, wu, wv, D;
    uu = dot(u,u);
    uv = dot(u,v);
    vv = dot(v,v);
    wu = dot(w,u);
    wv = dot(w,v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float t1, t2;
    t1 = (uv * wv - vv * wu) / D;
    if (t1 < -eps || t1 > 1.0 + eps)        // I is outside T
        return false;
    t2 = (uv * wu - uu * wv) / D;
    if (t2 < -eps || (t1+t2) > 1.0 + eps)  // I is outside T
        return false;

    return true;                      // I is in T
}

__global__ void identifySpecialBoundarySegments (uf4 *pos, ui4 *ep, int nvert, int nbe, uf4 *sbpos, ui4 *sbep, int sbnbe, float eps, int *sbid, int sbi){
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  uf4 vb[3];
  uf4 spos, norm;
  while(id < nbe){
    spos = pos[nvert+id];
    for(int i=0; i<sbnbe; i++){
      for(int j=0; j<3; j++)
        vb[j] = sbpos[sbep[i].a[j]];
      for(int j=0; j<3; j++)
        norm.a[j] = (vb[1].a[(j+1)%3]-vb[0].a[(j+1)%3])*(vb[2].a[(j+2)%3]-vb[0].a[(j+2)%3])
                  - (vb[1].a[(j+2)%3]-vb[0].a[(j+2)%3])*(vb[2].a[(j+1)%3]-vb[0].a[(j+1)%3]);
      if(segInTri(vb,spos,norm,eps)){
        sbid[nvert+id] = sbi;
        for(int j=0; j<3; j++)
          atomicAdd(&sbid[ep[id].a[j]],-1);
        break;
      }
    }
    id += blockDim.x*gridDim.x;
  }
  return;
}

__global__ void identifySpecialBoundaryVertices (int *sbid, int i, int *trisize, int nvert){

  int id = threadIdx.x + blockIdx.x*blockDim.x;
  while(id < nvert){
    if(-sbid[id] == trisize[id])
      sbid[id] = i;
    else if(sbid[id] < 0)
      sbid[id] = 0;
    id += blockDim.x*gridDim.x;
  }
  return;
}

// This function loops over all segments and checks whether all segments that are part of a
// special boundary with id sbi has at least one vertex that also has the same id. This is
// required for particle deletion at open boundaries as there needs to be at least one vertex
// to which the mass can be redistributed. If no vertex is found a warning is thrown.
__global__ void checkForSingularSegments (uf4 *pos, ui4 *ep, int nvert, int nbe, int *sbid, int sbi){
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  while(id < nbe){
    if(sbid[nvert + id] == sbi){
      int sameVert = 0;
      for(int i=0; i<3; i++){
        if(sbid[ep[id].a[i]] == sbi)
          sameVert += 1;
      }
      if(sameVert == 0)
        printf("-- WARNING -- segment at position (%lf,%lf,%lf) has no vertex of same special boundary id %d\n",
          pos[nvert+id].a[0], pos[nvert+id].a[1], pos[nvert+id].a[2], sbi);
    }

    id += blockDim.x*gridDim.x;
  }
  return;
}

__device__ bool segInTri(uf4 *vb, uf4 spos, uf4 norm, float eps){
  float dot00, dot01, dot02, dot11, dot12, invdet, u, v;
  dot00 = 0.;
  for(int i=0; i<3; i++)
    dot00 += norm.a[i]*(spos.a[i]-vb[0].a[i]);
  if(fabs(dot00) > eps)
    return false;
  dot00 = dot01 = dot02 = dot11 = dot12 = 0.;
  for(int i=0; i<3; i++){
    float ba = (vb[1].a[i] - vb[0].a[i]);
    float ca = (vb[2].a[i] - vb[0].a[i]);
    float pa = ( spos.a[i] - vb[0].a[i]);
    dot00 += ba*ba;
    dot01 += ba*ca;
    dot02 += ba*pa;
    dot11 += ca*ca;
    dot12 += ca*pa;
  }
  invdet = 1.0/(dot00*dot11-dot01*dot01);
  u = (dot11*dot02-dot01*dot12)*invdet;
  v = (dot00*dot12-dot01*dot02)*invdet;
  if( u < -eps || v < -eps || u+v > 1+eps ) return false;
  return true;
}

#endif
