/***********************************\
 *
 * TODO LIST:
 * - filling of complex geometries
 *
\***********************************/

#ifndef CRIXUS_CU
#define CRIXUS_CU

#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <hdf5.h>
#include <cuda.h>
#include "cuda_local.cuh"
#include "crixus.h"
#include "return.h"
#include "crixus_d.cuh"
#include "lock.cuh"

//#define bdebug 1

using namespace std;

int crixus_main(int argc, char** argv){
	//host
	cout << endl;
	cout << "\t*********************************" << endl;
	cout << "\t*                               *" << endl;
	cout << "\t*          C R I X U S          *" << endl;
	cout << "\t*                               *" << endl;
	cout << "\t*********************************" << endl;
	cout << "\t* Version: 0.3b                 *" << endl;
	cout << "\t* Date   : 16.11.2011           *" << endl;
	cout << "\t* Authors: Arno Mayrhofer       *" << endl;
	cout << "\t*          Christophe Kassiotis *"<< endl;
	cout << "\t*          F-X Morel            *"<< endl;
	cout << "\t*          Martin Ferrand       *"<< endl;
	cout << "\t*********************************" << endl;
	cout << endl;
	float m_v_floats[12];
	unsigned int through;
	short attribute;
	unsigned int num_of_facets;

  if(argc==1){
		cout << "No file specified." << endl;
		cout << "Correct use: crixus filename dr" << endl;
		cout << "Example use: crixus box.stl 0.1" << endl;
		return NO_FILE;
	}
	else if(argc==2){
		cout << "No particle discretization specified." << endl;
		cout << "Correct use: crixus filename dr" << endl;
		cout << "Example use: crixus box.stl 0.1" << endl;
		return NO_DR;
	}

	cout << "Opening file " << argv[1] << " ...";
	ifstream stl_file (argv[1], ios::in);
	if(!stl_file.is_open()){
		cout << " [FAILED]" << endl;
	  return FILE_NOT_OPEN;
	}
	cout << " [OK]" << endl;

	cout << "Checking whether stl file is not ASCII ...";
	bool issolid = true;
	char header[6] = "solid";
	for (int i=0; i<5; i++){
		char dum;
		stl_file.read((char *)&dum, sizeof(char));
		if(dum!=header[i]){
			issolid = false;
			break;
		}
	}
	if(issolid){
		cout << " [FAILED]" << endl;
		stl_file.close();
		return STL_NOT_BINARY;
	}
	stl_file.close();
	cout << " [OK]" << endl;

	// reopen file in binary mode
	stl_file.open(argv[1], ios::in | ios::binary);

	// read header
	for (int i=0; i<20; i++){
		float dum;
		stl_file.read((char *)&dum, sizeof(float));
	}
	// get number of facets
	stl_file.read((char *)&num_of_facets, sizeof(int));
	cout << "Reading " << num_of_facets << " facets ...";

	float dr = strtod(argv[2],NULL);
	// define variables
	vector< vector<float> > pos;
	vector< vector<float> > norm;
	vector< vector<float> >::iterator it;
	vector< vector<unsigned int> > epv;
	unsigned int nvert, nbe;
	vector<unsigned int> idum;
	vector<float> ddum;
	int iduma[3];
	for(int i=0;i<3;i++){
		ddum.push_back(0.);
		idum.push_back(0);
	}

	// read data
	through = 0;
	float xmin = 1e10, xmax = -1e10;
	float ymin = 1e10, ymax = -1e10;
	float zmin = 1e10, zmax = -1e10;
	while ((through < num_of_facets) & (!stl_file.eof()))
	{
		for (int i=0; i<12; i++)
		{
			stl_file.read((char *)&m_v_floats[i], sizeof(float));
		}
		for(int i=0;i<3;i++) ddum[i] = (float)m_v_floats[i];
		norm.push_back(ddum);
		for(int j=0;j<3;j++){
			for(int i=0;i<3;i++) ddum[i] = (float)m_v_floats[i+3*(j+1)];
			int k = 0;
			bool found = false;
			for(it = pos.begin(); it < pos.end(); it++){
				float diff = 0;
				for(int i=0;i<3;i++) diff += pow((*it)[i]-ddum[i],2);
				diff = sqrt(diff);
				if(diff < 1e-5*dr){
					idum[j] = k;
					found = true;
					break;
				}
				k++;
			}
			if(!found){
				pos.push_back(ddum);
				xmin = (xmin > ddum[0]) ? ddum[0] : xmin;
				xmax = (xmax < ddum[0]) ? ddum[0] : xmax;
				ymin = (ymin > ddum[1]) ? ddum[1] : ymin;
				ymax = (ymax < ddum[1]) ? ddum[1] : ymax;
				zmin = (zmin > ddum[2]) ? ddum[2] : zmin;
				zmax = (zmax < ddum[2]) ? ddum[2] : zmax;
				idum[j] = k;
			}
		}
		epv.push_back(idum);
		stl_file.read((char *)&attribute, sizeof(short));
		through++;
	}
	stl_file.close();
	if(num_of_facets != norm.size()){
		cout << " [FAILED]" << endl;
		return READ_ERROR;
	}
	nvert = pos.size();
	nbe   = norm.size();
	//create and copy vectors to arrays
	uf4 *norma, *posa;
	float *vola, *surf;
	ui4 *ep;
	norma = new uf4   [nbe];
	posa  = new uf4   [nvert+nbe];
	vola  = new float [nvert];
	surf  = new float [nbe]; //AM-TODO: could go to norma[3]
	ep    = new ui4   [nbe];
	for(unsigned int i=0; i<nvert+nbe; i++){
		if(i<nbe){
      for(int j=0; j<3; j++){
		  	norma[i].a[j] = norm[i][j];
		  	ep[i].a[j] = epv[i][j];
      }
		}
		if(i<nvert){
      for(int j=0; j<3; j++)
			  posa[i].a[j] = pos[i][j];
			vola[i] = 0.;
		}
	}
	//cuda arrays
	uf4 *norm_d;
	uf4 *pos_d;
	float *surf_d;
	ui4 *ep_d;
	CUDA_SAFE_CALL( cudaMalloc((void **) &norm_d,        nbe*sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &pos_d ,(nvert+nbe)*sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &surf_d,        nbe*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &ep_d  ,        nbe*sizeof(ui4  )) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) norm_d,(void *) norma,         nbe*sizeof(uf4  ), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) pos_d ,(void *) posa , (nvert+nbe)*sizeof(uf4  ), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) surf_d,(void *) surf ,         nbe*sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) ep_d  ,(void *) ep   ,         nbe*sizeof(ui4  ), cudaMemcpyHostToDevice) );
	cout << " [OK]" << endl;
	cout << "\n\tInformation:" << endl;
	cout << "\tOrigin of domain:           \t(" << xmin << ", " << ymin << ", " << zmin << ")\n";
	cout << "\tSize of domain:             \t(" << xmax-xmin << ", " << ymax-ymin << ", " << zmax-zmin << ")\n";
	cout << "\tNumber of vertices:         \t" << nvert << endl;
	cout << "\tNumber of boundary elements:\t" << nbe << "\n\n";

	//calculate surface and position of boundary elements
	cout << "Calculating surface and position of boundary elements ...";
  //device
	int numThreads, numBlocks;
	numThreads = threadsPerBlock;
	numBlocks = (int) ceil((float)nbe/(float)numThreads);
	numBlocks = min(numBlocks,50000);
	Lock lock;
	float xminp = 1e10, xminn = 1e10;
	float nminp = 0., nminn = 0.;
	float *xminp_d, *xminn_d;
	float *nminp_d, *nminn_d;
	CUDA_SAFE_CALL( cudaMalloc((void **) &xminp_d, sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &xminn_d, sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &nminp_d, sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &nminn_d, sizeof(float)) );
	CUDA_SAFE_CALL( cudaMemcpy(xminp_d, &xminp, sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(xminn_d, &xminn, sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(nminp_d, &nminp, sizeof(float), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(nminn_d, &nminn, sizeof(float), cudaMemcpyHostToDevice) );

	set_bound_elem<<<numBlocks, numThreads>>> (pos_d, norm_d, surf_d, ep_d, nbe, xminp_d, xminn_d, nminp_d, nminn_d, lock, nvert);

	CUDA_SAFE_CALL( cudaMemcpy((void *) posa,(void *) pos_d  , (nvert+nbe)*sizeof(uf4  ), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL(	cudaMemcpy((void *) surf,(void *) surf_d ,         nbe*sizeof(float), cudaMemcpyDeviceToHost) );
	//host
	for(unsigned int i=nvert; i<nvert+nbe; i++){
    for(int j=0; j<3; j++)
		  ddum[j] = posa[i].a[j];
		pos.push_back(ddum);
	}
	cout << " [OK]" << endl;
	cout << "\n\tNormals information:" << endl;
	cout << "\tPositive (n.(0,0,1)) minimum z: " << xminp << " (" << nminp << ")\n";
	cout << "\tNegative (n.(0,0,1)) minimum z: " << xminn << " (" << nminn << ")\n\n";
	char cont= 'n';
	do{
		if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
		cout << "Swap normals (y/n): ";
		cin >> cont;
	}while(cont!='y' && cont!='n');
	if(cont=='y'){
		cout << "Swapping normals ..."; //could be done on the device but probably isn't necessary
		for(unsigned int i=0; i<nbe; i++){
      for(int j=0; j<3; j++){
			  norm[i][j]    *= -1.;
			  norma[i].a[j] *= -1.;
      }
		}
		cout << " [OK]" << endl;
	}

	//calculate volume of vertex particles
	float dmin[3] = {xmin,ymin,zmin};
	float dmax[3] = {xmax,ymax,zmax};
	bool per[3] = {false, false, false};
	int *newlink;
	newlink = new int[nvert];
	for(unsigned int i=0; i<nvert; i++) newlink[i] = -1;
	for(unsigned int idim=0; idim<3; idim++){
		cont='n';
		do{
			if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
			if(idim==0){
				cout << "X-periodicity (y/n): ";
			}
			else if(idim==1){
				cout << "Y-periodicity (y/n): ";
			}
			else if(idim==2){
				cout << "Z-periodicity (y/n): ";
			}
			cin >> cont;
		}while(cont!='y' && cont!='n');
		if(cont=='y'){
			per[idim] = true;
			cout << "Updating links ...";
			//device
			for(unsigned int i=0; i<nvert; i++){
				if(fabs(posa[i].a[idim]-dmax[idim])<1e-5*dr){
					for(unsigned int j=0; j<nvert; j++){
						if(j==i) continue;
						if(sqrt(pow(posa[i].a[(idim+1)%3]-posa[j].a[(idim+1)%3],2.)+ \
						        pow(posa[i].a[(idim+2)%3]-posa[j].a[(idim+2)%3],2.)+ \
										pow(posa[j].a[idim      ]-dmin[idim]             ,2.) ) < 1e-4*dr){
							newlink[i] = j;
							break;
						}
						if(j==nvert-1){
							cout << " [FAILED]" << endl;
							return NO_PER_VERT;
						}
					}
				}
			}
			for(unsigned int i=0; i<nbe; i++){
        for(int j=0; j<3; j++){
				  if(newlink[ep[i].a[j]] != -1)
            ep[i].a[j] = newlink[ep[i].a[j]];
        }
			}
			unsigned int onvert = nvert;
			for(int i=onvert-1; i>=0; i--){
				if(newlink[i] != -1){
					for(unsigned int j=i+1; j<nvert+nbe; j++){
            for(int k=0; k<3; k++)
						  posa[i-1].a[k] = posa[i].a[k];
					}
					nvert--;
					for(unsigned int j=0; j<nbe; j++){
						for(unsigned int k=0; k<3; k++){
							if(ep[j].a[k]==i){
								cout << " [FAILED]" << endl;
								return NO_UNLINK;
							}
							if(ep[j].a[k]>i)
                ep[j].a[k]--;
						}
					}
				}
			}
			//host
			cout << " [OK]" << endl;
		} 
	}
	if(nvert+nbe != pos.size()){
		pos.clear();
		for(unsigned int i=0; i<nvert+nbe; i++){ //AM: posa should be made smaller as well, but not strictly necessary.
      for(int j=0; j<3; j++)
			  ddum[j] = posa[i].a[j];
			pos.push_back(ddum);
		}
	}
	cout << "Calculating volume of vertex particles ...";
	//device
	//get neighbouring vertices
	unsigned int *trisize;
	trisize = new unsigned int[nvert];
	for(unsigned int i=0; i<nvert; i++) trisize[i] = 0;
	for(unsigned int i=0; i<nbe; i++){
		for(unsigned int j=0; j<3; j++){
			trisize[ep[i].a[j]] += 1;
		}
	}
	//sort neighbouring vertices
	//calculate volume (geometry factor)
	const unsigned int gres = 10; // = dr/dr_grid
	const unsigned int gsize = gres*2+1; //makes sure that grid is large enough
	const float gdr = dr/(float)gres;
	float vgrid;
	float ***cvec;
	float avnorm[3];
	bool *first;
#ifdef bdebug
	vector <vector <float> > debug;
	vector <float> debug2;
#endif
	float vnorm;
	float eps=gdr*1e-4;
	bool closed;
	for(unsigned int i=0; i<nvert; i++){
		//initialize variables
		closed = true;
		int **tri;
		unsigned int tris = trisize[i];
		cvec = new float**[tris];
		tri = new int*[tris];
		first = new bool[tris];
		for(unsigned int j=0; j<tris; j++){
			first[j] = true;
			tri[j] = new int[3];
			cvec[j] = new float*[12];
			for(unsigned int k=0; k<12; k++) cvec[j][k] = new float[3];
		}
		for(unsigned int j=0; j<3; j++) avnorm[j] = 0.;
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
		vola[i] = 0.;
		float sp;
		//start big loop over all numerical integration points
		for(unsigned int k=0; k<gsize; k++){
		for(unsigned int l=0; l<gsize; l++){
		for(unsigned int m=0; m<gsize; m++){
			int ik = k-(gsize-1)/2;
			int il = l-(gsize-1)/2;
			int im = m-(gsize-1)/2;
			float px = ((float)ik)*gdr;
			float py = ((float)il)*gdr;
			float pz = ((float)im)*gdr;
			vgrid = 0.;
			for(unsigned int j=0; j<tris; j++){
				//create cubes
				if(k+l+m==0){
					//setting up cube directions
					for(unsigned int n=0; n<3; n++) cvec[j][2][n] = norma[tri[j][2]].a[n]; //normal of boundary element
					vnorm = 0.;
					for(unsigned int n=0; n<3; n++){
						cvec[j][0][n] = posa[tri[j][0]].a[n]-posa[i].a[n]; //edge 1
						if(per[n]&&fabs(cvec[j][0][n])>2*dr)	cvec[j][0][n] += sgn(cvec[j][0][n])*(-dmax[n]+dmin[n]); //periodicity
						vnorm += pow(cvec[j][0][n],2);
					}
					vnorm = sqrt(vnorm);
					for(unsigned int n=0; n<3; n++) cvec[j][0][n] /= vnorm; 
					for(unsigned int n=0; n<3; n++)	cvec[j][1][n] = cvec[j][0][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][0][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge1
					vnorm = 0.;
					for(unsigned int n=0; n<3; n++){
						cvec[j][3][n] = posa[tri[j][1]].a[n]-posa[i].a[n]; //edge 2
						if(per[n]&&fabs(cvec[j][3][n])>2*dr)	cvec[j][3][n] += sgn(cvec[j][3][n])*(-dmax[n]+dmin[n]); //periodicity
						vnorm += pow(cvec[j][3][n],2);
						avnorm[n] -= norma[tri[j][2]].a[n];
					}
					vnorm = sqrt(vnorm);
					for(unsigned int n=0; n<3; n++) cvec[j][3][n] /= vnorm; 
					for(unsigned int n=0; n<3; n++)	cvec[j][4][n] = cvec[j][3][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][3][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge2
				}
				//filling vgrid
				bool incube[5] = {false, false, false, false, false};
				for(unsigned int n=0; n<5; n++){
					sp = px*cvec[j][n][0]+py*cvec[j][n][1]+pz*cvec[j][n][2];
					if(fabs(sp)<=dr/2.+eps) incube[n] = true;
				}
				if((incube[0] && incube[1] && incube[2]) || (incube[2] && incube[3] && incube[4])){
					vgrid = 1.;
#ifdef bdebug
					if(i==0){ // && j==tris-1){
						ddum[0] = px+posa[i][0];
						ddum[1] = py+posa[i][1];
						ddum[2] = pz+posa[i][2];
						//cout << ddum[0] << " " << ddum[1] << " " << ddum[2] << endl;
						debug.push_back(ddum);
						debug2.push_back(vgrid);
					}
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
						cvec[j][5][n] = posa[tri[j][0]].a[n]-posa[i].a[n]; //normal of plane voronoi
						if(per[n]&&fabs(cvec[j][5][n])>2*dr)	cvec[j][5][n] += sgn(cvec[j][5][n])*(-dmax[n]+dmin[n]); //periodicity
						cvec[j][6][n] = posa[i].a[n]+cvec[j][5][n]/2.; //position of plane voronoi
						tvec[0][n] = cvec[j][5][n]; // edge 1
						tvec[1][n] = posa[tri[j][1]].a[n]-posa[i].a[n]; // edge 2
						if(per[n]&&fabs(tvec[1][n])>2*dr)	tvec[1][n] += sgn(tvec[1][n])*(-dmax[n]+dmin[n]); //periodicity
						if(!closed){
							cvec[j][7][n] = tvec[1][n]; //normal of plane voronoi 2
							cvec[j][8][n] = posa[i].a[n]+cvec[j][7][n]/2.; //position of plane voronoi 2
						}
						tvec[2][n] = avnorm[n]; // negative average normal
					}
					for(unsigned int n=0; n<3; n++){
						for(unsigned int k=0; k<3; k++){
							cvec[j][k+9][n] = tvec[k][(n+1)%3]*tvec[(k+1)%3][(n+2)%3]-tvec[k][(n+2)%3]*tvec[(k+1)%3][(n+1)%3]; //normals of tetrahedron planes
						}
					}
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += norma[tri[j][2]].a[n]*cvec[j][9][n]; //test whether normals point inward tetrahedron, if no flip normals
					if(sp > 0.){
						for(unsigned int k=0; k<3; k++){
							for(unsigned int n=0; n<3; n++)	cvec[j][k+9][n] *= -1.;
						}
					}
				}
			  //remove unwanted points and sum up for volume
				//voronoi plane
				tvec[0][0] = px + posa[i].a[0] - cvec[j][6][0];
				tvec[0][1] = py + posa[i].a[1] - cvec[j][6][1];
				tvec[0][2] = pz + posa[i].a[2] - cvec[j][6][2];
				sp = 0.;
				for(unsigned int n=0; n<3; n++) sp += tvec[0][n]*cvec[j][5][n];
				if(sp>0.+eps){
					vgrid = 0.;
					break;
				}
				else if(fabs(sp) < eps){
					vgrid /= 2.;
				}
				//voronoi plane 2
				if(!closed){
					tvec[0][0] = px + posa[i].a[0] - cvec[j][8][0];
					tvec[0][1] = py + posa[i].a[1] - cvec[j][8][1];
					tvec[0][2] = pz + posa[i].a[2] - cvec[j][8][2];
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
				tvec[0][0] = px;
				tvec[0][1] = py;
				tvec[0][2] = pz;
				bool half = false;
				for(unsigned int o=0; o<3; o++){
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += tvec[0][n]*cvec[j][9+o][n];
					if(sp<0.-eps) break;
					if(fabs(sp)<eps && o==0) half=true;
					if(o==2 && !half){
						vgrid = 0.;
						break;
					}
					else if(o==2 && half){
						vgrid /= 2.;
					}
				}
				if(vgrid < eps) break;
				if(j==tris-1)	vola[i] += vgrid;
			}
		}
		}
		}
		vola[i] *= pow(dr/(float)gres,3);
		delete [] first;
		for(unsigned int j=0; j<tris; j++){
			for(unsigned int k=0; k<12; k++) delete [] cvec[j][k];
			delete [] cvec[j];
			delete [] tri[j];
		}
		delete [] cvec;
		delete [] tri;
	}
	//host
	vector <float> vol;
	for(unsigned int i=0; i<nvert; i++) vol.push_back(vola[i]);
	cout << " [OK]" << endl;

	//setting up fluid particles
	cout << "Defining fluid particles in a box ..." << endl;
	bool set = true;
	float **fpos;
	float fluid_vol = pow(dr,3);
	unsigned int nfluid = 0;
	unsigned int maxf = 0;
	eps = 1e-4*dr;
	while(set){
		xmin = xmax = ymin = ymax = zmin = zmax = 0.;
		cout << "Enter dimensions of fluid box:" << endl;
		cout << "xmin, xmax: ";
		cin >> xmin >> xmax;
		cout << "ymin, ymax: ";
		cin >> ymin >> ymax;
		cout << "zmin, zmax: ";
		cin >> zmin >> zmax;
		if(fabs(xmin-xmax)<1e-5*dr || fabs(ymin-ymax)<1e-5*dr || fabs(zmin-zmax)<1e-5*dr){
			cout << "\nMistake in input for fluid box dimensions" << endl;
			cout << "Fluid particle definition ... [FAILED]" << endl;
			return FLUID_NDEF;
		}
		maxf = (floor((xmax+eps-xmin)/dr)+1)*(floor((ymax+eps-ymin)/dr)+1)*(floor((zmax+eps-zmin)/dr)+1);
		fpos = new float*[maxf];
		for(unsigned int i=0; i<maxf; i++) fpos[i] = new float[3];
		//device
		//this can be a bit more complex in order to fill complex geometries
		for(int i=0; i<=floor((xmax+eps-xmin)/dr); i++){
		for(int j=0; j<=floor((ymax+eps-ymin)/dr); j++){
		for(int k=0; k<=floor((zmax+eps-zmin)/dr); k++){
			fpos[nfluid][0] = xmin + (float)i*dr;
			fpos[nfluid][1] = ymin + (float)j*dr;
			fpos[nfluid][2] = zmin + (float)k*dr;
			nfluid++;
		}}}
		//host
		for(unsigned int i=0; i<nfluid; i++){
			ddum[0] = fpos[i][0];
			ddum[1] = fpos[i][1];
			ddum[2] = fpos[i][2];
			pos.push_back(ddum);
			vol.push_back(fluid_vol);
		}
		cont = 'n';
		do{
			if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
			cout << "Another fluid box (y/n): ";
			cin >> cont;
			if(cont=='n') set = false;
		}while(cont!='y' && cont!='n');
	}
	cout << "Creation of " << nfluid << " fluid particles completed. [OK]" << endl;

	//prepare output structure
	cout << "Creating and initializing of output buffer ...";
	OutBuf *buf;
#ifndef bdebug
	buf = new OutBuf[pos.size()];
#else
	buf = new OutBuf[pos.size()+debug.size()];
#endif
	int k=0;
	//fluid particles
	for(unsigned int i=nbe+nvert; i<pos.size(); i++){
		buf[k].x = pos[i][0];
		buf[k].y = pos[i][1];
		buf[k].z = pos[i][2];
		buf[k].nx = 0.;
		buf[k].ny = 0.;
		buf[k].nz = 0.;
		buf[k].vol = vol[i-nbe];
		buf[k].surf = 0.;
		buf[k].kpar = 1;
		buf[k].kfluid = 1;
		buf[k].kent = 1;
		buf[k].kparmob = 0;
		buf[k].iref = k;
		buf[k].ep1 = 0;
		buf[k].ep2 = 0;
		buf[k].ep3 = 0;
		k++;
	}
	//vertex particles
	for(unsigned int i=0; i<nvert; i++){
		buf[k].x = pos[i][0];
		buf[k].y = pos[i][1];
		buf[k].z = pos[i][2];
		buf[k].nx = 0.;
		buf[k].ny = 0.;
		buf[k].nz = 0.;
		buf[k].vol = vol[i];
		buf[k].surf = 0.;
		buf[k].kpar = 2;
		buf[k].kfluid = 1;
		buf[k].kent = 1;
		buf[k].kparmob = 0;
		buf[k].iref = k;
		buf[k].ep1 = 0;
		buf[k].ep2 = 0;
		buf[k].ep3 = 0;
		k++;
	}
	//boundary elements
	for(unsigned int i=nvert; i<nvert+nbe; i++){
		buf[k].x = pos[i][0];
		buf[k].y = pos[i][1];
		buf[k].z = pos[i][2];
		buf[k].nx = norm[i-nvert][0];
		buf[k].ny = norm[i-nvert][1];
		buf[k].nz = norm[i-nvert][2];
		buf[k].vol = 0.;
		buf[k].surf = surf[i-nvert];
		buf[k].kpar = 3;
		buf[k].kfluid = 1;
		buf[k].kent = 1;
		buf[k].kparmob = 0;
		buf[k].iref = k;
		buf[k].ep1 = nfluid+ep[i-nvert].a[0]; //AM-TODO: maybe + 1 as indices in fortran start with 1
		buf[k].ep2 = nfluid+ep[i-nvert].a[1];
		buf[k].ep3 = nfluid+ep[i-nvert].a[2];
		k++;
	}
#ifdef bdebug
	//debug
	for(unsigned int i=0; i<debug.size(); i++){
		buf[k].x = debug[i][0];
		buf[k].y = debug[i][1];
		buf[k].z = debug[i][2];
		pos.push_back(debug[i]);
		buf[k].nx = 0;
		buf[k].ny = 0;
		buf[k].nz = 0;
		buf[k].vol = debug2[i];
		buf[k].surf = 0.;
		buf[k].kpar = 4;
		buf[k].kfluid = 1;
		buf[k].kent = 1;
		buf[k].kparmob = 0;
		buf[k].iref = k;
		buf[k].ep1 = 0;
		buf[k].ep2 = 0;
		buf[k].ep3 = 0;
		k++;
	}
#endif
	cout << " [OK]" << endl;

	//Output
	int flen = strlen(argv[1]);
	char *fname = new char[flen+5];
	const char *fend = "h5sph";
	float time = 0.;
	fname[0] = '0';
	fname[1] = '.';
	strncpy(fname+2, argv[1], flen-3);
	strncpy(fname+flen-1, fend, strlen(fend));
	fname[flen+4] = '\0';
	cout << "Writing output to file " << fname << " ...";
	int err = hdf5_output( buf, pos.size(), fname, &time);
	if(err==0){ cout << " [OK]" << endl; }
	else {
		cout << " [FAILED]" << endl;
		return WRITE_FAIL;
	}
	return 0;
}

int hdf5_output (OutBuf *buf, int len, const char *filename, float *timevalue){
	hid_t		mem_type_id, loc_id, dataset_id, file_space_id, mem_space_id, xfer_plist_id;
	hsize_t	count[1], offset[1], dim[] = {len};
	herr_t	status;

	xfer_plist_id = H5Pcreate(H5P_FILE_ACCESS);
	loc_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, xfer_plist_id);
	H5Pclose(xfer_plist_id);
	file_space_id = H5Screate_simple(1, dim, NULL);
	mem_type_id = H5Tcreate(H5T_COMPOUND, sizeof(OutBuf));

	H5Tinsert(mem_type_id, "Coords_0"       , HOFFSET(OutBuf, x),       H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Coords_1"       , HOFFSET(OutBuf, y),       H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Coords_2"       , HOFFSET(OutBuf, z),       H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Normal_0"       , HOFFSET(OutBuf, nx),      H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Normal_1"       , HOFFSET(OutBuf, ny),      H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Normal_2"       , HOFFSET(OutBuf, nz),      H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Volume"         , HOFFSET(OutBuf, vol),     H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Surface"        , HOFFSET(OutBuf, surf),    H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "ParticleType"   , HOFFSET(OutBuf, kpar),    H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "FluidType"      , HOFFSET(OutBuf, kfluid),  H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "KENT"           , HOFFSET(OutBuf, kent),    H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "MovingBoundary" , HOFFSET(OutBuf, kparmob), H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "AbsoluteIndex"  , HOFFSET(OutBuf, iref),    H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "VertexParticle1", HOFFSET(OutBuf, ep1),     H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "VertexParticle2", HOFFSET(OutBuf, ep2),     H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "VertexParticle3", HOFFSET(OutBuf, ep3),     H5T_NATIVE_INT);

	dataset_id = H5Dcreate(loc_id, DATASETNAME, mem_type_id, file_space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(file_space_id);

	count[0] = len;
	offset[0] = 0;
	mem_space_id = H5Screate_simple(1, count, NULL);
	file_space_id = H5Dget_space(dataset_id);
	H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
	xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
	status = H5Dwrite(dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buf);
	if(status < 0) return status;

	H5Dclose(dataset_id);
	H5Sclose(file_space_id);
	H5Sclose(mem_space_id);
	H5Pclose(xfer_plist_id);
	H5Fclose(loc_id);
	H5Tclose(mem_type_id);

	return 0;
}

#endif
