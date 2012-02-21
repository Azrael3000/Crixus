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

using namespace std;

int crixus_main(int argc, char** argv){
	//host
	cout << endl;
	cout << "\t*********************************" << endl;
	cout << "\t*                               *" << endl;
	cout << "\t*          C R I X U S          *" << endl;
	cout << "\t*                               *" << endl;
	cout << "\t*********************************" << endl;
	cout << "\t* Version: 0.4b                 *" << endl;
	cout << "\t* Date   : 09.02.2012           *" << endl;
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
	
	//looking for cuda devices without timeout
	cout << "Selecting GPU ...";
	int dcount, maxblock, maxthread;
	size_t globMemSize;
	maxthread = threadsPerBlock;
	bool found = false;
	CUDA_SAFE_CALL( cudaGetDeviceCount(&dcount) );
	for (int i=0; i<dcount; i++){
		cudaDeviceProp prop;
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop,i) );
		if(!prop.kernelExecTimeoutEnabled){
			found = true;
			CUDA_SAFE_CALL( cudaSetDevice(i) );
			maxthread = prop.maxThreadsPerBlock;
			maxblock  = prop.maxGridSize[0];
			globMemSize = prop.totalGlobalMem;
			cout << " Id: " << i << " (" << maxthread << ", " << maxblock << ") ...";
			if(maxthread < threadsPerBlock){
				cout << " [FAILED]" << endl;
				return MAXTHREAD_TOO_BIG;
			}
			cout << " [OK]" << endl;
			break;
		}
	}
	if(!found){
		cudaDeviceProp prop;
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop,0) );
		maxthread = prop.maxThreadsPerBlock;
		if(maxthread < threadsPerBlock){
			cout << " [FAILED]" << endl;
			return MAXTHREAD_TOO_BIG;
		}
		cout << " [OK]" << endl;
		cout << "\n\tWARNING:" << endl;
		cout << "\tCould not find GPU without timeout." << endl;
		cout << "\tIf execution terminates with timeout reduce gres.\n" << endl;
	}
	if(maxthread != threadsPerBlock){
		cout << "\n\tINFORMATION:" << endl;
		cout << "\tthreadsPerBlock is not equal to maximum number of available threads.\n" << endl;
	}

	//Reading file
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
	for(unsigned int i=0; i<max(nvert,nbe); i++){
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
	int numThreads, numBlocks;
	numThreads = threadsPerBlock;
	numBlocks = (int) ceil((float)nbe/(float)numThreads);
	numBlocks = min(numBlocks,maxblock);
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
	CUDA_SAFE_CALL( cudaMemcpy(&xminp, xminp_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(&xminn, xminn_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(&nminp, nminp_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(&nminn, nminn_d, sizeof(float), cudaMemcpyDeviceToHost) );
	cudaFree (xminp_d);
	cudaFree (xminn_d);
	cudaFree (nminp_d);
	cudaFree (nminn_d);
	//host
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
		cout << "Swapping normals ...";
		
		swap_normals<<<numBlocks, numThreads>>> (norm_d, nbe);

	  CUDA_SAFE_CALL( cudaMemcpy((void *) norma,(void *) norm_d, nbe*sizeof(uf4), cudaMemcpyDeviceToHost) );

		cout << " [OK]" << endl;
	}
	cout << endl;

	//calculate volume of vertex particles
	uf4 dmin = {xmin,ymin,zmin,0.};
	uf4 dmax = {xmax,ymax,zmax,0.};
	bool per[3] = {false, false, false};
	int *newlink, *newlink_h;
	uf4 *dmin_d, *dmax_d;
	newlink_h = new int[nvert];
	CUDA_SAFE_CALL( cudaMalloc((void **) &newlink, nvert*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &dmin_d ,       sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &dmax_d ,       sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) dmin_d , (void *) &dmin    ,       sizeof(float4), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) dmax_d , (void *) &dmax    ,       sizeof(float4), cudaMemcpyHostToDevice) );
	for(unsigned int idim=0; idim<3; idim++){
		cont='n';
		do{
			if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
			if(idim==0){
				cout << "X-periodicity (y/n): "; }
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
	for(int i=0; i<nvert; i++)
		newlink_h[i] = -1;
	CUDA_SAFE_CALL( cudaMemcpy((void *) newlink, (void *) newlink_h, nvert*sizeof(int)   , cudaMemcpyHostToDevice) );
			numBlocks = (int) ceil((float)max(nvert,nbe)/(float)numThreads);
			numBlocks = min(numBlocks,maxblock);

			find_links <<<numBlocks, numThreads>>> (pos_d, nvert, dmax_d, dmin_d, dr, newlink, idim);
			periodicity_links<<<numBlocks,numThreads>>>(pos_d, ep_d, nvert, nbe, dmax_d, dmin_d, dr, newlink, idim);

			CUDA_SAFE_CALL( cudaMemcpy((void *) posa,(void *) pos_d, (nvert+nbe)*sizeof(uf4), cudaMemcpyDeviceToHost) );
			CUDA_SAFE_CALL( cudaMemcpy((void *) ep  ,(void *) ep_d ,         nbe*sizeof(ui4), cudaMemcpyDeviceToHost) );
			//if(err!=0) return err;
			//host
			cout << " [OK]" << endl;
		} 
	}
	CUDA_SAFE_CALL( cudaFree(newlink) );

	cout << "\nCalculating volume of vertex particles ...";
	float eps=dr/(float)gres*1e-4;
	int *trisize, *trisize_h;
	float *vol_d;
  bool *per_d;
	trisize_h = new int[nvert];
	for(int i=0; i<nvert; i++)
		trisize_h[i] = 0;
	CUDA_SAFE_CALL( cudaMalloc((void **) &trisize, nvert*sizeof(int  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &vol_d  , nvert*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &per_d  ,     3*sizeof(bool )) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) per_d  , (void *) per      ,     3*sizeof(bool), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) trisize, (void *) trisize_h, nvert*sizeof(int) , cudaMemcpyHostToDevice) );
	numBlocks = (int) ceil((float)nvert/(float)numThreads);
	numBlocks = min(numBlocks,maxblock);

	calc_trisize <<<numBlocks, numThreads>>> (ep_d, trisize, nbe);
#ifndef bdebug
	calc_vert_volume <<<numBlocks, numThreads>>> (pos_d, norm_d, ep_d, vol_d, trisize, dmin_d, dmax_d, nvert, nbe, dr, eps, per_d);
#else
	uf4 *debug, *debug_d;
	int debugs = pow((gres*2+1),3);
	float *debugp, *debugp_d;
	debugp = new float [100];
	debug = new uf4[debugs];
	CUDA_SAFE_CALL( cudaMalloc((void **) &debug_d, debugs*sizeof(uf4)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &debugp_d, 100*sizeof(float)) );

	calc_vert_volume <<<numBlocks, numThreads>>> (pos_d, norm_d, ep_d, vol_d, trisize, dmin_d, dmax_d, nvert, nbe, dr, eps, per_d, debug_d, debugp_d);

	CUDA_SAFE_CALL( cudaMemcpy((void*) debug, (void*) debug_d, debugs*sizeof(uf4), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void*) debugp, (void*) debugp_d, 100*sizeof(float), cudaMemcpyDeviceToHost) );
	for(int i=0; i<20; i++){
		cout << i << " " << debugp[i] << endl;
	}
#endif

	CUDA_SAFE_CALL( cudaMemcpy((void *) vola,(void *) vol_d, nvert*sizeof(float), cudaMemcpyDeviceToHost) );
	cudaFree( trisize );
	cudaFree( vol_d   );

	cout << " [OK]" << endl;

	//gathering information for grid generation
	//note that the domainsize should be at least 2*dr greater than the actual domain.
	cout << "\nDefinition of computational domain:" << endl;
	cout << "Proposed domainsize:" << endl;
	for(int i=0; i<3; i++){
		if(!per[i]){
			dmin.a[i] -= dr;
			dmax.a[i] += dr;
		}
		cout << "\tmin and max in " << ((i==0)?"x":((i==1)?"y":"z")) << "-direction: " << dmin.a[i] << " " << dmax.a[i] << endl;
	}
	cont= 'n';
	do{
		if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
	  cout << "Accept proposed domain size (y/n):";
		cin >> cont;
	}while(cont!='y' && cont!='n');
	if(cont=='n'){
		for(int i=0; i<3; i++){
			if(per[i]){
				cout << "" << ((i==0)?"x":((i==1)?"y":"z")) << "-direction set due to periodicity." << endl;
				cout << " min and max in " << ((i==0)?"x":((i==1)?"y":"z")) << "-direction: " << dmin.a[i] << " " << dmax.a[i] << endl;
			}
			else{
				cout << "Please enter min and max for " << ((i==0)?"x":((i==1)?"y":"z")) << "-direction: ";
				cin >> dmin.a[i] >> dmax.a[i];
			}
		}
	}
	float hdr;
	cout << "\nEnter ratio h/dr: ";
	cin >> hdr;
	int iker=0;
	float krad=0.;
	cout << "\n\tKernel list:" << endl;
	cout << "\t1 - Wendland kernel" << endl;
	cout << "\nChoose kernel by entering the associated number: ";
	cin >> iker;
	if(iker<=0 || iker>1) return WRONG_INPUT;
	switch(iker){
	case 1:
	default:
		krad = 2*hdr*dr;
	}
	
	/*
	//getting number of gridpoints
	cout << "\nCalculating number of grid points ...";
	int *ngridp_d;
	int ngridp=0;
	unsigned int *igrid;
	unsigned int *igrid_d;
	eps = 1e-4*dr;
	int maxgridp = floor((dmax.a[0]+eps-dmin.a[0])/dr+1.)*floor((dmax.a[1]+eps-dmin.a[1])/dr+1.)*floor((dmax.a[2]+eps-dmin.a[2])/dr+1.);
	int igridsbyte = (int)ceil((float)maxgridp/8.); //bytes required for bitfield
	int igrids = (int)ceil((float)igridsbyte/(float)sizeof(unsigned int)); //# of unsigned ints required for bitfield

	igrid = new unsigned int[igrids];
	for(int i=0; i<igrids; i++)
		igrid[i] = 0;
	CUDA_SAFE_CALL( cudaMalloc((void **) &ngridp_d,        sizeof(int         )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &igrid_d , igrids*sizeof(unsigned int)) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) ngridp_d, (void *) &ngridp,        sizeof(int         ), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) dmin_d  , (void *) &dmin  ,        sizeof(float4      ), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) dmax_d  , (void *) &dmax  ,        sizeof(float4      ), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) igrid_d , (void *) igrid  , igrids*sizeof(unsigned int), cudaMemcpyHostToDevice) );
	numBlocks = (int) ceil((float)maxgridp/(float)numThreads);
	numBlocks = min(numBlocks,maxblock);
	Lock lock_ngridp;

	calc_ngridp <<<numBlocks,numThreads>>> (pos_d, igrid_d, dmin_d, dmax_d, per_d, ngridp_d, maxgridp, dr, eps, nvert, nbe, krad, lock_ngridp, igrids);

	CUDA_SAFE_CALL( cudaMemcpy((void *) &ngridp, (void *) ngridp_d,        sizeof(int         ), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) igrid  , (void *) igrid_d , igrids*sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaFree( ngridp_d );
	cudaFree( igrid_d  );
	cout << " (" << ngridp << ")";
	cout << " [OK]" << endl;

	//initializing grid positions
	cout << "Initializing grid points ...";
	uf4   *gpos , *gpos_d;   // position of grid point and index
	float *gam  , *gam_d;    // gamma of grid point
	float *ggam , *ggam_d;   // grad gamma_{pb} of grid point
	int   *iggam, *iggam_d;  // index of boundary element
	int   nrggam, *nrggam_d; // number of links of all grid points
	int   *ilock;            // lock while calculating gamma
	gpos  = new uf4  [ngridp];
	gam   = new float[ngridp];
	ggam  = new float[ngridp*maxlink*3];
	iggam = new int[ngridp*maxlink];
	CUDA_SAFE_CALL( cudaMalloc((void **) &gpos_d  ,           ngridp*sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &gam_d   ,           ngridp*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &ilock   ,           ngridp*sizeof(int  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &ggam_d  , ngridp*maxlink*3*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &iggam_d ,   ngridp*maxlink*sizeof(int  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &nrggam_d,                  sizeof(int  )) );
	//setting indices
	const unsigned int uibs = 8*sizeof(unsigned int);
	unsigned int byte[uibs];
	for(int i=0; i<uibs; i++)
		byte[i] = 1<<i;
	int gid = 0;
	for(int i=0; i<maxgridp; i++){
		int ida = i/uibs;
		int idi = i%uibs;
		if((byte[idi] & igrid[ida]) != 0){
			gpos[gid].a[3] = (float)i; //gpos[].a[3] is the unique grid point index
			gid += 1;
			if(gid > ngridp){
				cout << " [FAILED]";
				return BITFIELD_WRONG;
			}
		}
	}
	delete [] igrid;
	nrggam = 0;
	CUDA_SAFE_CALL( cudaMemcpy((void *) gpos_d  , (void *)  gpos  , ngridp*sizeof(uf4), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) nrggam_d, (void *) &nrggam,        sizeof(int), cudaMemcpyHostToDevice) );
	numBlocks = (int) ceil((float)ngridp/(float)numThreads);
	numBlocks = min(numBlocks,maxblock);
	float *deb, *deb_d;
	deb = new float[numBlocks*numThreads];
	CUDA_SAFE_CALL( cudaMalloc((void **) &deb_d, numBlocks*numThreads*sizeof(float)) );
	Lock lock_gpoints;
	float seed = 0.5; //insert time here
	
	init_gpoints <<<numBlocks,numThreads>>> (pos_d, ep_d, surf_d, norm_d, gpos_d, gam_d, ggam_d, iggam_d, dmin_d, dmax_d, per_d, ngridp, dr, hdr, iker, eps, nvert, nbe, krad, seed, nrggam_d, lock_gpoints, deb_d, ilock);

	CUDA_SAFE_CALL( cudaMemcpy((void *) deb   , (void *) deb_d  ,           numBlocks*numThreads*sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) gpos   , (void *) gpos_d  ,           ngridp*sizeof(uf4  ), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) gam    , (void *) gam_d   ,           ngridp*sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) ggam   , (void *) ggam_d  , ngridp*maxlink*3*sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) iggam  , (void *) iggam_d ,   ngridp*maxlink*sizeof(int  ), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) &nrggam, (void *) nrggam_d,                  sizeof(int  ), cudaMemcpyDeviceToHost) );
//	for(int i=0; i<numBlocks*numThreads; i++) if(deb[i] > 0.5) cout << i << " " <<  deb[i] << endl;
//	for(int i=0; i<24; i+= 3)  cout << deb[i] << " " << deb[i+1] << " " << deb[i+2] << endl;

	cudaFree(deb_d   );
	cudaFree(gpos_d  );
	cudaFree(ggam_d  );
	cudaFree(iggam_d );
	cudaFree(nrggam_d);
	cout << " [OK]" << endl; */
	cudaFree(norm_d  );
	cudaFree(pos_d   );
	cudaFree(vol_d   );
	cudaFree(surf_d  );
	cudaFree(ep_d    );

	//setting up fluid particles
	cout << "\nDefining fluid particles in a box ..." << endl;
	bool set = true;
	float fluid_vol = pow(dr,3);
	unsigned int nfluid = 0;
	unsigned int *nfluid_in_box;
	unsigned int nfbox = 0;
	unsigned int maxf = 0, maxfn;
	uf4 **fpos;
	fpos = new uf4*[maxfbox];
	nfluid_in_box = new unsigned int[maxfbox];
	for(int i=0; i<maxfbox; i++)
		nfluid_in_box[i] = 0;
	int *nfib_d;
	CUDA_SAFE_CALL( cudaMalloc((void **) &nfib_d, sizeof(int)) );

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
		maxfn = (floor((xmax+eps-xmin)/dr)+1)*(floor((ymax+eps-ymin)/dr)+1)*(floor((zmax+eps-zmin)/dr)+1);
		size_t memReq = (size_t)maxfn*sizeof(uf4);
		int it = (int)ceil((float)memReq/0.6/(float)globMemSize); // use at most 60% of the memory
		if(nfbox + it >= maxfbox){
			cout << "Too many fluid boxes " << nfbox << " " << it << endl;
			break;
		}
		int ixpos = 0;
		int nxplane = (floor((xmax+eps-xmin)/dr)+1);
		for(int i=0; i<it; i++){
			//1D domain decomposition
			int ixplane = nxplane/it + ((i<nxplane%it)?1:0);
			maxf = ixplane*(floor((ymax+eps-ymin)/dr)+1)*(floor((zmax+eps-zmin)/dr)+1);
			float xmin_l = xmin + (float)ixpos*dr;
			ixpos += ixplane;
			float xmax_l = xmin + (float)(ixpos-1)*dr;
			fpos[nfbox] = new uf4[maxf];
			uf4 *fpos_d;
			int nfib;
			nfib = 0;
			CUDA_SAFE_CALL( cudaMalloc((void **) &fpos_d, maxf*sizeof(uf4)) );
			CUDA_SAFE_CALL( cudaMemcpy((void *) nfib_d, (void *) &nfib, sizeof(int), cudaMemcpyHostToDevice) );
			numBlocks = (int) ceil((float)maxf/(float)numThreads);
			numBlocks = min(numBlocks,maxblock);

			Lock lock_f;
			fill_fluid<<<numBlocks, numThreads>>> (fpos_d, xmin_l, xmax_l, ymin, ymax, zmin, zmax, eps, dr, nfib_d, maxf, lock_f);
			
			CUDA_SAFE_CALL( cudaMemcpy((void *) fpos[nfbox], (void *) fpos_d, maxf*sizeof(uf4), cudaMemcpyDeviceToHost) );
			CUDA_SAFE_CALL( cudaMemcpy((void *) &nfib      , (void *) nfib_d,      sizeof(int), cudaMemcpyDeviceToHost) );
			cudaFree( fpos_d );
			nfluid_in_box[nfbox] = nfib;
			nfluid += nfib;
			nfbox += 1;
		}

		cont = 'n';
		if(nfbox == maxfbox)
			break;
		do{
			if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
			cout << "Another fluid box (y/n): ";
			cin >> cont;
			if(cont=='n') set = false;
		}while(cont!='y' && cont!='n');
	}
	cout << "\nCreation of " << nfluid << " fluid particles completed. [OK]" << endl;
	cudaFree( nfib_d );

	//prepare output structure for particles
	cout << "Creating and initializing of output buffer of particles ...";
	OutBuf *buf;
#ifndef bdebug
	unsigned int nelem = nvert+nbe+nfluid;
#else
	unsigned int nelem = nvert+nbe+nfluid+debugs;
#endif
	buf = new OutBuf[nelem];
	int k=0;
	//fluid particles
	for(unsigned int j=0; j<nfbox; j++){
		for(unsigned int i=0; i<nfluid_in_box[j]; i++){
			if(fpos[j][i].a[0] < -1e9)
				continue;
			buf[k].x = fpos[j][i].a[0];
			buf[k].y = fpos[j][i].a[1];
			buf[k].z = fpos[j][i].a[2];
			buf[k].nx = 0.;
			buf[k].ny = 0.;
			buf[k].nz = 0.;
			buf[k].vol = fluid_vol;
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
	}
	//vertex particles
	for(unsigned int i=0; i<nvert; i++){
		if(posa[i].a[0] < -1e9){
			nelem--;
			continue;
		}
		buf[k].x = posa[i].a[0];
		buf[k].y = posa[i].a[1];
		buf[k].z = posa[i].a[2];
		buf[k].nx = 0.;
		buf[k].ny = 0.;
		buf[k].nz = 0.;
		buf[k].vol = vola[i];
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
		buf[k].x = posa[i].a[0];
		buf[k].y = posa[i].a[1];
		buf[k].z = posa[i].a[2];
		buf[k].nx = norma[i-nvert].a[0];
		buf[k].ny = norma[i-nvert].a[1];
		buf[k].nz = norma[i-nvert].a[2];
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
	for(unsigned int i=0; i<debugs; i++){
		buf[k].x = debug[i].a[0];
		buf[k].y = debug[i].a[1];
		buf[k].z = debug[i].a[2];
		buf[k].nx = 0;
		buf[k].ny = 0;
		buf[k].nz = 0;
		buf[k].vol = debug[i].a[3];
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

	//Output of particles
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
	int err = hdf5_output( buf, nelem, fname, &time);
	if(err==0){ cout << " [OK]" << endl; }
	else {
		cout << " [FAILED]" << endl;
		return WRITE_FAIL;
	}

	/*
	//Preparing output of gridpoints
	cout << "Creating and initializing of output buffer of grid points ...";
	gOutBuf *gbuf;
	linkOutBuf *linkbuf;
	gbuf = new gOutBuf[ngridp];
	linkbuf = new linkOutBuf[nrggam];
	k=0;
	for(int i=0; i<ngridp; i++){
		gbuf[i].x   = gpos[i].a[0];
		gbuf[i].y   = gpos[i].a[1];
		gbuf[i].z   = gpos[i].a[2];
		gbuf[i].id  = (int) gpos[i].a[3];
		gbuf[i].gam = gam[i];
		gbuf[i].ggamx = 0.;
		gbuf[i].ggamy = 0.;
		gbuf[i].ggamz = 0.;
		for(int j=0; j<maxlink; j++){
			if(ggam[i*maxlink*3+j*3] < -1e9)
				break;
			if(k<nrggam){
			linkbuf[k].id    = gbuf[i].id;
			linkbuf[k].iggam = iggam[i*maxlink+j];
			linkbuf[k].ggamx = ggam[i*maxlink*3+j*3];
			linkbuf[k].ggamy = ggam[i*maxlink*3+j*3+1];
			linkbuf[k].ggamz = ggam[i*maxlink*3+j*3+2];
			gbuf[i].ggamx += linkbuf[k].ggamx;
			gbuf[i].ggamy += linkbuf[k].ggamy;
			gbuf[i].ggamz += linkbuf[k].ggamz;
			k++;
			}
			if(k>nrggam){
				cout << " [FAILED]" << endl;
				return NRGGAM_WRONG;
			}
		}
	}
	if(k!=nrggam){
	 cout << " [FAILED]" << endl;
	 return NRGGAM_WRONG;
	}	
	cout << " [OK]" << endl;

	//Output of gridpoints
	delete [] fname;
	fname = new char[flen+10];
	fname[0] = '0';
	fname[1] = '.';
	fname[2] = 'g';
	fname[3] = 'r';
	fname[4] = 'i';
	fname[5] = 'd';
	fname[6] = '.';
	strncpy(fname+7, argv[1], flen-3);
	strncpy(fname+flen+4, fend, strlen(fend));
	fname[flen+9] = '\0';
	cout << "Writing grid to file " << fname << " ...";
	err = hdf5_grid_output( gbuf, ngridp, fname, &time);
	if(err==0){ cout << " [OK]" << endl; }
	else {
		cout << " [FAILED]" << endl;
		return WRITE_FAIL;
	}

	//Output of gridpoint ggam_{pb}
	delete [] fname;
	fname = new char[flen+10];
	fname[0] = '0';
	fname[1] = '.';
	fname[2] = 'l';
	fname[3] = 'i';
	fname[4] = 'n';
	fname[5] = 'k';
	fname[6] = '.';
	strncpy(fname+7, argv[1], flen-3);
	strncpy(fname+flen+4, fend, strlen(fend));
	fname[flen+9] = '\0';
	cout << "Writing grid ggam links to file " << fname << " ...";
	err = hdf5_link_output( linkbuf, nrggam, fname, &time);
	if(err==0){ cout << " [OK]" << endl; }
	else {
		cout << " [FAILED]" << endl;
		return WRITE_FAIL;
	}
	*/

	//Free memory
	//Arrays
	delete [] norma;
	delete [] posa;
	delete [] vola;
	delete [] surf;
	delete [] ep;
	delete [] nfluid_in_box;
	delete [] buf;
	delete [] fname;
	for(unsigned int i=0; i<nfbox; i++)
		delete [] fpos[i];
	delete [] fpos;
	/*
	delete [] ggam;
	delete [] iggam;
	delete [] gpos;
	*/
	//Cuda
	cudaFree( per_d   );
	cudaFree( dmin_d  );
	cudaFree( dmax_d  );

	//End
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

int hdf5_grid_output (gOutBuf *buf, int len, const char *filename, float *timevalue){
	hid_t		mem_type_id, loc_id, dataset_id, file_space_id, mem_space_id, xfer_plist_id;
	hsize_t	count[1], offset[1], dim[] = {len};
	herr_t	status;

	xfer_plist_id = H5Pcreate(H5P_FILE_ACCESS);
	loc_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, xfer_plist_id);
	H5Pclose(xfer_plist_id);
	file_space_id = H5Screate_simple(1, dim, NULL);
	mem_type_id = H5Tcreate(H5T_COMPOUND, sizeof(gOutBuf));

	H5Tinsert(mem_type_id, "Coords_0"       , HOFFSET(gOutBuf, x),       H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Coords_1"       , HOFFSET(gOutBuf, y),       H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Coords_2"       , HOFFSET(gOutBuf, z),       H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Gamma"          , HOFFSET(gOutBuf, gam),     H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "Index"          , HOFFSET(gOutBuf, id),      H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "grad_gamma_0"   , HOFFSET(gOutBuf, ggamx),   H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "grad_gamma_1"   , HOFFSET(gOutBuf, ggamy),   H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "grad_gamma_2"   , HOFFSET(gOutBuf, ggamz),   H5T_NATIVE_FLOAT);

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

int hdf5_link_output (linkOutBuf *buf, int len, const char *filename, float *timevalue){
	hid_t		mem_type_id, loc_id, dataset_id, file_space_id, mem_space_id, xfer_plist_id;
	hsize_t	count[1], offset[1], dim[] = {len};
	herr_t	status;

	xfer_plist_id = H5Pcreate(H5P_FILE_ACCESS);
	loc_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, xfer_plist_id);
	H5Pclose(xfer_plist_id);
	file_space_id = H5Screate_simple(1, dim, NULL);
	mem_type_id = H5Tcreate(H5T_COMPOUND, sizeof(linkOutBuf));

	H5Tinsert(mem_type_id, "Index"          , HOFFSET(linkOutBuf, id),      H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "boundary_index" , HOFFSET(linkOutBuf, iggam),   H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "grad_gamma_0"   , HOFFSET(linkOutBuf, ggamx),   H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "grad_gamma_1"   , HOFFSET(linkOutBuf, ggamy),   H5T_NATIVE_FLOAT);
	H5Tinsert(mem_type_id, "grad_gamma_2"   , HOFFSET(linkOutBuf, ggamz),   H5T_NATIVE_FLOAT);

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
