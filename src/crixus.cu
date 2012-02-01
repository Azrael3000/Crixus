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
	CUDA_SAFE_CALL( cudaMemcpy(&xminp, xminp_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(&xminn, xminn_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(&nminp, nminp_d, sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(&nminn, nminn_d, sizeof(float), cudaMemcpyDeviceToHost) );
	cudaFree (xminp_d)
	cudaFree (xminn_d)
	cudaFree (nminp_d)
	cudaFree (nminn_d)
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

	//calculate volume of vertex particles
	float4 dmin = {xmin,ymin,zmin,0.};
	float4 dmax = {xmax,ymax,zmax,0.};
	bool per[3] = {false, false, false};
	int *newlink;
	uf4 *dmin_d, dmax_d;
	CUDA_SAFE_CALL( cudaMalloc((void **) &newlink, nvert*sizeof(float)) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &dmin_d ,       sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &dmax_d ,       sizeof(uf4  )) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) dmin_d, (void *) dmin, sizeof(float4), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy((void *) dmax_d, (void *) dmax, sizeof(float4), cudaMemcpyHostToDevice) );
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
			numBlocks = (int) ceil((float)max(nvert,nbe)/(float)numThreads);
			numBlocks = min(numBlocks,50000);

			//init for gpu sync
			int *sync_id, *sync_od;
			initsync(sync_id, sync_od, numBlocks);

			int err = periodicity_links<<<numBlocks,numThreads>>>(pos_d, ep_d, nvert, nbe, dmax_d, dmin_d, dr, sync_id, sync_od, newlink, idim);

			CUDA_SAFE_CALL( cudaMemcpy((void *) posa,(void *) pos_d, (nvert+nbe)*sizeof(uf4), cudaMemcpyDeviceToHost) );
			CUDA_SAFE_CALL( cudaMemcpy((void *) ep  ,(void *) ep_d ,         nbe*sizeof(ui4), cudaMemcpyHostToDevice) );
			if(err!=0) return err;
			//host
			cout << " [OK]" << endl;
		} 
	}
	CUDA_SAFE_CALL( cudaFree(newlink) );

	cout << "Calculating volume of vertex particles ...";
	int *trisize;
	float *vol_d;
	CUDA_SAFE_CALL( cudaMalloc((void **) &trisize, nvert*sizeof(int  )) );
	CUDA_SAFE_CALL( cudaMalloc((void **) &vol_d  ,       sizeof(float)) );
	numBlocks = (int) ceil((float)max(nvert)/(float)numThreads);
	numBlocks = min(numBlocks,50000);

	calc_vert_volume <<<numBlocks, numThreads>>> (pos_d, norm_d, ep_d, vol_d, trisize, dmin_d, dmax_d sync_i, sync_o, nvert, nbe, dr, eps);

	CUDA_SAFE_CALL( cudaMemcpy((void *) vola,(void *) vol_d, nvert*sizeof(float), cudaMemcpyDeviceToHost) );
	cudaFree( trisize );
	cudaFree( vol_d   );

	cout << " [OK]" << endl;

	//setting up fluid particles
	cout << "Defining fluid particles in a box ..." << endl;
	bool set = true;
	float fluid_vol = pow(dr,3);
	unsigned int nfluid = 0;
	unsigned int *nfluid_in_box;
	unsigned int nfbox = 0;
	unsigned int maxf = 0;
	eps = 1e-4*dr;
	uf4 **fpos;
	fpos = new uf4*[maxfbox];
	nfluid_in_box = new int[maxfbox];
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
		maxf = (floor((xmax+eps-xmin)/dr)+1)*(floor((ymax+eps-ymin)/dr)+1)*(floor((zmax+eps-zmin)/dr)+1);
		fpos[nfbox] = new uf4[maxf];
		uf4 *fpos_d;
		int nfib;
		nfib = 0;
		CUDA_SAFE_CALL( cudaMalloc((void **) &fpos_d, maxf*sizeof(uf4)) );
		CUDA_SAFE_CALL( cudaMemcpy((void *) nfib_d, (void *) nfib, sizeof(int), cudaMemcpyHostToDevice) );
		numBlocks = (int) ceil((float)maxf/(float)numThreads);
		numBlocks = min(numBlocks,50000);

		fill_fluid<<<numBlocks, numThreads>>> (fpos_d, xmin, xmax, ymin, ymax, zmin, zmax, eps, dr, nfib_d, fmax, lock);
		
		CUDA_SAFE_CALL( cudaMemcpy((void *) fpos[nfbox], (void *) fpos_d, maxf*sizeof(uf4), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaMemcpy((void *) nfib       , (void *) nfib_d,      sizeof(int), cudaMemcpyDeviceToHost) );
		cudaFree( fpos_d );
		nfluid_in_box[nfbox] = nfib;
		nfluid += nfib;
		cudaFree(fpos_d);
		nfbox += 1;

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
	cout << "Creation of " << nfluid << " fluid particles completed. [OK]" << endl;
	cudaFree( nfib_d );

	//prepare output structure
	cout << "Creating and initializing of output buffer ...";
	OutBuf *buf;
#ifndef bdebug
	buf = new OutBuf[nvert + nbe + nfluid];
#else
	buf = new OutBuf[nvert + nbe + nfluid + debug.size()];
#endif
	int k=0;
	//fluid particles
	for(unsigned int j=0; j<nfbox; j++){
		for(unsigned int i=0; i<nfuid_in_box[j]; i++){
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
		if(posa[i].a[0] < -1e9)
			continue;
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
	int err = hdf5_output( buf, sizeof(buf)/sizeof(OutBuf), fname, &time);
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
	delete [] nfluid_in_box;
	delete [] buf;
	delete [] fname;
	for(unsigned int i=0; i<maxfbox; i++)
		delete [] fpos[i];
	delete [] fpos;
	//Cuda
	cudaFree( norm_d  );
	cudaFree( pos_d   );
	cudaFree( vol_d   );
	cudaFree( surf_d  );
	cudaFree( ep_d    );
	cudaFree( dmin_d  );
	cudaFree( dmax_d  );
	cudaFree( sync_id );
	cudaFree( sync_od );

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

#endif
