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

  //periodicity
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
  delete [] newlink_h;

	//calculate volume of vertex particles
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
	for(int i=0; i<30; i++){
		cout << i << " " << debugp[i] << endl;
	}
#endif

	CUDA_SAFE_CALL( cudaMemcpy((void *) vola,(void *) vol_d, nvert*sizeof(float), cudaMemcpyDeviceToHost) );
	//cout << endl;
	//for(int i=0; i<30; i++) cout << i << " " << vola[i] << endl;
	cudaFree( trisize );
	cudaFree( vol_d   );

	cout << " [OK]" << endl;

	cudaFree(vol_d   );
	cudaFree(surf_d  );

	//setting up fluid particles
	cout << "\nDefining fluid particles ..." << endl;

	cout << "Checking wether coarse grid is available ...";
	bool bcoarse;
	int flen = strlen(argv[1]);
	char *cfname = new char[flen+7];
	strncpy(cfname, argv[1], flen-4);
	cfname[flen-4] = '_';
	cfname[flen-3] = 'c';
	cfname[flen-2] = 'o';
	cfname[flen-1] = 'a';
	cfname[flen-0] = 'r';
	cfname[flen+1] = 's';
	cfname[flen+2] = 'e';
	strncpy(cfname+flen+3, argv[1]+flen-4, 4);
	stl_file.open(cfname, ios::in | ios::binary); //no check is done whether or not the file is actually binary or not
	if(!stl_file.is_open()){
		bcoarse = false;
		cout << " [NO]" << endl;
	}
	else{
		bcoarse = true;
		cout << " [YES]" << endl;
	}

	cout << "Checking wether fluid geometry is available ...";
	bool bfgeom;
	char *ffname = new char[flen+7];
	strncpy(cfname, argv[1], flen-4);
	cfname[flen-4] = '_';
	cfname[flen-3] = 'f';
	cfname[flen-2] = 's';
	cfname[flen-1] = 'h';
	cfname[flen-0] = 'a';
	cfname[flen+1] = 'p';
	cfname[flen+2] = 'e';
	strncpy(cfname+flen+3, argv[1]+flen-4, 4);
	ifstream fstl_file (ffname, ios::in | ios::binary); //no check is done whether or not the file is actually binary or not
	if(!fstl_file.is_open()){
		bfgeom = false;
		cout << " [NO]" << endl;
	}
	else{
		bfgeom = true;
		cout << " [YES]" << endl;
	}

	bool set = true;
	bool firstfgeom = true;
	unsigned int cnvert, cnbe;
	uf4 *cnorma, *cposa;
	ui4 *cep;
	unsigned int nfluid = 0;
	unsigned int nfbox = 0;
	unsigned int maxf = 0, maxfn;
	int opt;
	unsigned int *fpos, *fpos_d;
  unsigned int *nfi_d;
	maxfn = int(floor((dmax.a[0]-dmin.a[0]+eps)/dr+1)*floor((dmax.a[1]-dmin.a[1]+eps)/dr+1)*floor((dmax.a[2]-dmin.a[2]+eps)/dr+1));
	maxf = int(ceil(float(maxfn)/8./float(sizeof(unsigned int))));
	fpos = new unsigned int [maxf];
	CUDA_SAFE_CALL( cudaMalloc((void **) &fpos_d, maxf*sizeof(unsigned int)) );

	while(set){
		xmin = xmax = ymin = ymax = zmin = zmax = 0.;
		if(bfgeom){
			cout << "Choose option:" << endl;
			cout << " 1 ... Fluid in a box" << endl;
			cout << " 2 ... Fluid based on geometry" << endl;
			cout << "Input: ";
			opt = 0;
			cin >> opt;
			while(opt<1 || opt>2){
				cout << "Wrong input try again: ";
				cin >> opt;
			}
		}
		else{
			opt = 1;
		}

		if(opt==1){ // fluid based on rectangular box
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
			numBlocks = (int) ceil((float)maxf/(float)numThreads);
			numBlocks = min(numBlocks,maxblock);

			Lock lock_f;
			unsigned int nfi;
			CUDA_SAFE_CALL( cudaMalloc((void **) &nfi_d, sizeof(unsigned int)) );
			fill_fluid<<<numBlocks, numThreads>>> (fpos_d, nfi_d, xmin, xmax, ymin, ymax, zmin, zmax, dmin_d, dmax_d, eps, dr, lock_f);
			CUDA_SAFE_CALL( cudaMemcpy((void *) &nfi, (void *) nfi_d, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
			nfluid += nfi;
				
		}

		else if(opt==2){ // fluid based on geometry
			// get seed point
			float spos[3];
			cout << "Please specify a seed point." << endl;
			cout << "x, y, z = ";
			cin >> spos[0] >> spos[1] >> spos[2];

			// initialize geometry if first run
			if(firstfgeom){
				firstfgeom = false;

				cudaFree(norm_d  );
				cudaFree(pos_d   );
				cudaFree(ep_d    );

				// if coarse grid for geometry is available read it
				if(bcoarse){
					// read header
					for (int i=0; i<20; i++){
						float dum;
						stl_file.read((char *)&dum, sizeof(float));
					}
					// get number of facets
					stl_file.read((char *)&num_of_facets, sizeof(int));
					cout << "Reading " << num_of_facets << " facets of coarse geometry ...";

					// define variables
					pos.clear();
					norm.clear();
					epv.clear();
					ddum.clear();
					idum.clear();
					for(int i=0;i<3;i++){
						ddum.push_back(0.);
						idum.push_back(0);
					}

					// read data
					through = 0;
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
					delete [] norma;
					delete [] posa;
					delete [] ep;
					//create and copy vectors to arrays
					norma = new uf4   [nbe];
					posa  = new uf4   [nvert+nbe];
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
						}
					}
					pos.clear();
          epv.clear();
					norm.clear();
				}

				// read fluid geometry
				// read header
				for (int i=0; i<20; i++){
				float dum;
				fstl_file.read((char *)&dum, sizeof(float));
				}
				// get number of facets
				fstl_file.read((char *)&num_of_facets, sizeof(int));
				cout << "Reading " << num_of_facets << " facets of fluid geometry ...";

				// define variables
				pos.clear();
				norm.clear();
				epv.clear();
				ddum.clear();
				idum.clear();
				for(int i=0;i<3;i++){
					ddum.push_back(0.);
					idum.push_back(0);
				}

				// read data
				through = 0;
				while ((through < num_of_facets) & (!fstl_file.eof()))
				{
					for (int i=0; i<12; i++){
						fstl_file.read((char *)&m_v_floats[i], sizeof(float));
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
							idum[j] = k;
						}
					}
					epv.push_back(idum);
					fstl_file.read((char *)&attribute, sizeof(short));
					through++;
				}
				fstl_file.close();
				if(num_of_facets != norm.size()){
					cout << " [FAILED]" << endl;
					return READ_ERROR;
				}
				cnvert = pos.size();
				cnbe   = norm.size();
				//create and copy vectors to arrays
				cnorma = new uf4   [nbe];
				cposa  = new uf4   [nvert];
				cep    = new ui4   [nbe];
				for(unsigned int i=0; i<max(nbe,nvert); i++){
					if(i<nbe){
						cnorma[i] = norma[i];
						cep   [i] = ep   [i];
					}
					if(i<nvert){
						cposa [i] = posa [i];
					}
				}
				delete [] norma;
				delete [] posa;
				delete [] ep;
				norma = new uf4   [nbe+cnbe];
				posa  = new uf4   [nvert+cnvert];
				ep    = new ui4   [nbe+cnbe];
				for(unsigned int i=0; i<max(nbe,nvert); i++){
					if(i<nbe){
						norma[i] = cnorma[i];
						ep   [i] = cep   [i];
					}
					if(i<nvert){
						posa [i] = cposa [i];
					}
				}
				delete [] cnorma;
				delete [] cposa;
				delete [] cep;
				for(unsigned int i=0; i<max(cnvert,cnbe); i++){
					if(i<cnbe){
						for(int j=0; j<3; j++){
							norma[i+nbe].a[j] = norm[i][j];
							ep[i+nbe].a[j] = epv[i][j];
						}
					}
					if(i<cnvert){
						for(int j=0; j<3; j++)
							posa[i+nvert].a[j] = pos[i][j];
					}
				}
				nvert += cnvert;
				nbe += cnbe;
				pos.clear();
				epv.clear();
				norm.clear();
				CUDA_SAFE_CALL( cudaMalloc((void **) &norm_d,   nbe*sizeof(uf4)) );
				CUDA_SAFE_CALL( cudaMalloc((void **) &pos_d , nvert*sizeof(uf4)) );
				CUDA_SAFE_CALL( cudaMalloc((void **) &ep_d  ,   nbe*sizeof(ui4)) );
				CUDA_SAFE_CALL( cudaMemcpy((void *) norm_d, (void *) norma,   nbe*sizeof(uf4), cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL( cudaMemcpy((void *) pos_d , (void *) posa , nvert*sizeof(uf4), cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL( cudaMemcpy((void *) ep_d  , (void *) ep   ,   nbe*sizeof(ui4), cudaMemcpyHostToDevice) );
			}
		}

		cont = 'n';
		if(nfbox == maxfbox)
			break;
		do{
			if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
			cout << "Another fluid container (y/n): ";
			cin >> cont;
			if(cont=='n') set = false;
		}while(cont!='y' && cont!='n');
	}
	CUDA_SAFE_CALL( cudaMemcpy((void *) fpos, (void *) fpos_d, maxf*sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cout << "\nCreation of " << nfluid << " fluid particles completed. [OK]" << endl;
	cudaFree( fpos_d );
	cudaFree( nfi_d );
	cudaFree(norm_d  );
	cudaFree(pos_d   );
	cudaFree(ep_d    );

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
	unsigned int m,n,imin[3];
  float fluid_vol = pow(dr,3);
	imin[0] = ceil((dmax.a[0]-dmin.a[0])/dr);
	imin[1] = ceil((dmax.a[1]-dmin.a[1])/dr);
	imin[2] = ceil((dmax.a[2]-dmin.a[2])/dr);
	//fluid particles
	for(unsigned int j=0; j<maxfn; j++){
		int i = j/8;
		int l = j%8;
		m = 1 << l;
		if(fpos[i] | m){
			m = maxfn/(imin[1]*imin[2]);
			buf[k].x = dmin.a[0]+dr*m;
			n = maxfn - m*(imin[1]*imin[2]);
			m = n/imin[2];
			buf[k].y = dmin.a[1]+dr*m;
			m = n - m*imin[2];
			buf[k].z = dmin.a[2]+dr*m;
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
  int *nvshift;
  nvshift = new int[nvert];
  for(unsigned int i=0; i<nvert; i++)
    nvshift[i] = 0;
  int ishift = 0;
	for(unsigned int i=0; i<nvert; i++){
		if(posa[i].a[0] < -1e9){
			nelem--;
      ishift++;
			continue;
		}
    nvshift[i] = ishift;
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
		buf[k].ep1 = nfluid+ep[i-nvert].a[0] - nvshift[ep[i-nvert].a[0]];
		buf[k].ep2 = nfluid+ep[i-nvert].a[1] - nvshift[ep[i-nvert].a[1]];
		buf[k].ep3 = nfluid+ep[i-nvert].a[2] - nvshift[ep[i-nvert].a[2]];
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

	//Free memory
	//Arrays
	delete [] norma;
	delete [] posa;
	delete [] vola;
	delete [] surf;
	delete [] ep;
	delete [] buf;
	delete [] fname;
	delete [] fpos;
	//Cuda
	cudaFree( per_d   );
	cudaFree( dmin_d  );
	cudaFree( dmax_d  );

	//End
	return 0;
}
#endif
