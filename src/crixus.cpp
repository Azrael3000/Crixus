/***********************************\
 *
 * TODO LIST:
 * - filling of complex geometries
 *
\***********************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <hdf5.h>
#include "crixus.h"
#include "return.h"

// #define debug 0

using namespace std;

int main(int argc, char* argv[]){
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

	double dr = strtod(argv[2],NULL);
	// define variables
	vector< vector<double> > pos;
	vector< vector<double> > norm;
	vector< vector<double> >::iterator it;
	vector< vector<int> > ep;
	unsigned int nvert, nbe;
	vector<int> idum;
	vector<double> ddum;
	int iduma[3];
	double dduma[3];
	for(int i=0;i<3;i++){
		ddum.push_back(0.);
		idum.push_back(0);
	}

	// read data
	through = 0;
	double xmin = 1e10, xmax = -1e10;
	double ymin = 1e10, ymax = -1e10;
	double zmin = 1e10, zmax = -1e10;
	while ((through < num_of_facets) & (!stl_file.eof()))
	{
		for (int i=0; i<12; i++)
		{
			stl_file.read((char *)&m_v_floats[i], sizeof(float));
		}
		for(int i=0;i<3;i++) ddum[i] = (double)m_v_floats[i];
		norm.push_back(ddum);
		for(int j=0;j<3;j++){
			for(int i=0;i<3;i++) ddum[i] = (double)m_v_floats[i+3*(j+1)];
			int k = 0;
			bool found = false;
			for(it = pos.begin(); it < pos.end(); it++){
				double diff = 0;
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
		ep.push_back(idum);
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
	double *norma[3], *posa[3];
	double *vola, *surf;
	for(unsigned int i=0; i<3; i++){
		norma[i] = new double[nbe];
		posa[i]  = new double[nvert];
	}
	vola = new double[nvert];
	surf = new double[nbe];
	for(unsigned int i=0; i<max(nvert,nbe); i++){
		if(i<nvert){
			posa[i][0]  = pos[i][0];
			posa[i][1]  = pos[i][1];
			posa[i][2]  = pos[i][2];
			vola[i]     = 0.;
		}
		if(i<nbe){
			norma[i][0] = norm[i][0];
			norma[i][1] = norm[i][1];
			norma[i][2] = norm[i][2];
		}
	}
	cout << " [OK]" << endl;
	cout << "\n\tInformation:" << endl;
	cout << "\tOrigin of domain:           \t(" << xmin << ", " << ymin << ", " << zmin << ")\n";
	cout << "\tSize of domain:             \t(" << xmax-xmin << ", " << ymax-ymin << ", " << zmax-zmin << ")\n";
	cout << "\tNumber of vertices:         \t" << nvert << endl;
	cout << "\tNumber of boundary elements:\t" << nbe << "\n\n";

	//calculate surface and position of boundary elements
	cout << "Calculating surface and position of boundary elements ...";
//	double *surf; //, *ta, *tb, *tc;
	double xminp = 1e10, xminn = 1e10;
	double nminp = 0., nminn = 0.;
/*	ta   = new double[nbe];
	tb   = new double[nbe];
	tc   = new double[nbe]; */
	for(unsigned int i=0; i<nbe; i++){
		//formula: a = 1/4 sqrt(4*a^2*b^2-(a^2+b^2-c^2)^2)
		double a2 = 0.;
		double b2 = 0.;
		double c2 = 0.;
		dduma[0] = 0.;
		dduma[1] = 0.;
		dduma[2] = 0.;
		for(int j=0; j<3; j++){
			dduma[j] += pos[ep[i][0]][j]/3.;
			dduma[j] += pos[ep[i][1]][j]/3.;
			dduma[j] += pos[ep[i][2]][j]/3.;
			a2 += pow(pos[ep[i][0]][j]-pos[ep[i][1]][j],2);
			b2 += pow(pos[ep[i][1]][j]-pos[ep[i][2]][j],2);
			c2 += pow(pos[ep[i][2]][j]-pos[ep[i][0]][j],2);
		}
		if(norm[i][2] > 1e-5 && xminp > dduma[2]){
			xminp = dduma[2];
			nminp = norm[i][2];
		}
		if(norm[i][2] < -1e-5 && xminn > dduma[2]){
			xminn = dduma[2];
			nminn = norm[i][2];
		}
		surf[i] = 0.25*sqrt(4.*a2*b2-pow(a2+b2-c2,2));
		posa[i+nvert][0] = dduma[0];
		posa[i+nvert][1] = dduma[1];
		posa[i+nvert][2] = dduma[2];
/*		ta[i] = sqrt(a2);
		tb[i] = sqrt(b2);
		tc[i] = sqrt(c2); */
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
		cout << "Swapping normals ...";
		for(unsigned int i=0; i<nbe; i++){
			norm[i][0] *= -1.;
			norm[i][1] *= -1.;
			norm[i][2] *= -1.;
		}
		cout << " [OK]" << endl;
	}

	//calculate volume of vertex particles
	double dmin[3] = {xmin,ymin,zmin};
	double dmax[3] = {xmax,ymax,zmax};
	bool per[3] = {false, false, false};
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
			vector <int> newlink (nvert,-1);
			for(unsigned int i=0; i<nvert; i++){
				if(fabs(pos[i][idim]-dmax[idim])<1e-5*dr){
					for(unsigned int j=0; j<nvert; j++){
						if(j==i) continue;
						if(sqrt(pow(pos[i][(idim+1)%3]-pos[j][(idim+1)%3],2.)+ \
						        pow(pos[i][(idim+2)%3]-pos[j][(idim+2)%3],2.)+ \
										pow(pos[j][idim]      -dmin[idim]        ,2.) ) < 1e-4*dr){
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
				for(unsigned int j=0; j<3; j++){
					if(newlink[ep[i][j]] != -1){ ep[i][j] = newlink[ep[i][j]]; }
				}
			}
			unsigned int onvert = nvert;
			for(int i=onvert-1; i>=0; i--){
				if(newlink[i] != -1){
					pos.erase(pos.begin()+i);
					nvert--;
					for(unsigned int j=0; j<nbe; j++){
						for(unsigned int k=0; k<3; k++){
							if(ep[j][k]==i){
								cout << " [FAILED]" << endl;
								return NO_UNLINK;
							}
							if(ep[j][k]>i){ ep[j][k]--; }
						}
					}
				}
			}
			cout << " [OK]" << endl;
		} 
	}
	cout << "Calculating volume of vertex particles ...";
	//get neighbouring vertices
	vector <vector <int> > *tri;
	tri = new vector <vector <int> >[nvert];
	for(unsigned int i=0; i<nbe; i++){
		for(unsigned int j=0; j<3; j++){
			idum[0] = ep[i][(j+1)%3];
			idum[1] = ep[i][(j+2)%3];
			idum[2] = i;
			tri[ep[i][j]].push_back(idum);
		}
	}
	//sort neighbouring vertices
	//calculate volume (geometry factor)
	const unsigned int gres = 50; // = dr/dr_grid
	const unsigned int gsize = gres*2+1; //makes sure that grid is large enough
	const double gdr = dr/(double)gres;
	double vgrid;
	vector <vector <vector <double> > > cvec;
	vector <double> avnorm;
	vector <bool> first;
#ifdef debug
	vector <vector <double> > debug;
	vector <double> debug2;
#endif
	double vnorm;
	double eps=gdr*1e-4;
	bool closed;
	for(unsigned int i=0; i<nvert; i++){
		closed = true;
		unsigned int tris = tri[i].size();
		cvec.resize(tris);
		avnorm.resize(3);
		first.resize(tris);
		for(unsigned int j=0; j<tris; j++){
			cvec[j].resize(12);
			if(j<3) avnorm[j] = 0.;
			first[j] = true;
			for(unsigned int k=0; k<12; k++){
				cvec[j][k].resize(3);
			}
		}
		for(unsigned int j=0; j<tris; j++){
			for(unsigned int k=j+1; k<tris; k++){
				if(tri[i][j][1] == tri[i][k][0]){
					if(k!=j+1){
						idum = tri[i][j+1];
						tri[i][j+1] = tri[i][k];
						tri[i][k] = idum;
					}
					break;
				}
				if(tri[i][j][1] == tri[i][k][1]){
					idum[0] = tri[i][k][1];
					idum[1] = tri[i][k][0];
					idum[2] = tri[i][k][2];
					tri[i][k] = tri[i][j+1];
					tri[i][j+1] = idum;
					break;
				}
				if(k==tris-1) closed = false;
			}
		}
		if(tri[i][0][0] != tri[i][tris-1][1]){
			closed = false;
		}
		vola[i] = 0.;
		double sp;
		for(unsigned int k=0; k<gsize; k++){
		for(unsigned int l=0; l<gsize; l++){
		for(unsigned int m=0; m<gsize; m++){
			int ik = k-(gsize-1)/2;
			int il = l-(gsize-1)/2;
			int im = m-(gsize-1)/2;
			double px = ((double)ik)*gdr;
			double py = ((double)il)*gdr;
			double pz = ((double)im)*gdr;
			vgrid = 0.;
			for(unsigned int j=0; j<tris; j++){
				//create cubes
				if(k+l+m==0){ //AM FIXME this only works if this is never set (vgrid = 0) which should most of the time be the case but is no good practise
					//setting up cube directions
					for(unsigned int n=0; n<3; n++) cvec[j][2][n] = norm[tri[i][j][2]][n]; //normal of boundary element
					vnorm = 0.;
					for(unsigned int n=0; n<3; n++){
						cvec[j][0][n] = pos[tri[i][j][0]][n]-pos[i][n]; //edge 1
						if(per[n]&&fabs(cvec[j][0][n])>2*dr)	cvec[j][0][n] += sgn(cvec[j][0][n])*(-dmax[n]+dmin[n]); //periodicty
						vnorm += pow(cvec[j][0][n],2);
					}
					vnorm = sqrt(vnorm);
					for(unsigned int n=0; n<3; n++) cvec[j][0][n] /= vnorm; 
					for(unsigned int n=0; n<3; n++)	cvec[j][1][n] = cvec[j][0][(n+1)%3]*cvec[j][2][(n+2)%3]-cvec[j][0][(n+2)%3]*cvec[j][2][(n+1)%3]; //cross product of normal and edge1
					vnorm = 0.;
					for(unsigned int n=0; n<3; n++){
						cvec[j][3][n] = pos[tri[i][j][1]][n]-pos[i][n]; //edge 2
						if(per[n]&&fabs(cvec[j][3][n])>2*dr)	cvec[j][3][n] += sgn(cvec[j][3][n])*(-dmax[n]+dmin[n]); //periodicty
						vnorm += pow(cvec[j][3][n],2);
						avnorm[n] -= norm[tri[i][j][2]][n];
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
					break;
				}
			}
			//end create cubes
			//remove points based on planes (voronoi diagram & walls)
			double tvec[3][3];
			for(unsigned int j=0; j<tris; j++){
				if(vgrid<eps) break; //gridpoint already empty
				if(first[j]){
					first[j] = false;
					//set up plane normals and points
					for(unsigned int n=0; n<3; n++){
						cvec[j][5][n] = pos[tri[i][j][0]][n]-pos[i][n]; //normal of plane voronoi
						if(per[n]&&fabs(cvec[j][5][n])>2*dr)	cvec[j][5][n] += sgn(cvec[j][5][n])*(-dmax[n]+dmin[n]); //periodicty
						cvec[j][6][n] = pos[i][n]+cvec[j][5][n]/2.; //position of plane voronoi
						tvec[0][n] = cvec[j][5][n]; // edge 1
						tvec[1][n] = pos[tri[i][j][1]][n]-pos[i][n]; // edge 2
						if(per[n]&&fabs(tvec[1][n])>2*dr)	tvec[1][n] += sgn(tvec[1][n])*(-dmax[n]+dmin[n]); //periodicty
						if(!closed){
							cvec[j][7][n] = tvec[1][n]; //normal of plane voronoi 2
							cvec[j][8][n] = pos[i][n]+cvec[j][7][n]/2.; //position of plane voronoi 2
						}
						tvec[2][n] = avnorm[n]; // negative average normal
					}
					for(unsigned int n=0; n<3; n++){
						for(unsigned int k=0; k<3; k++){
							cvec[j][k+9][n] = tvec[k][(n+1)%3]*tvec[(k+1)%3][(n+2)%3]-tvec[k][(n+2)%3]*tvec[(k+1)%3][(n+1)%3]; //normals of tetrahedron planes
							//if(i==48) cout << cvec[j][k+9][n] << " " << n << " " << k << endl;
						}
					}
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += norm[tri[i][j][2]][n]*cvec[j][9][n]; //test whether normals point inward tetrahedron, if no flip normals
					if(sp > 0.){
						for(unsigned int k=0; k<3; k++){
							for(unsigned int n=0; n<3; n++)	cvec[j][k+9][n] *= -1.;
						}
					}
				}
			  //remove unwanted points and sum up for volume
				//voronoi plane
				tvec[0][0] = px + pos[i][0] - cvec[j][6][0];
				tvec[0][1] = py + pos[i][1] - cvec[j][6][1];
				tvec[0][2] = pz + pos[i][2] - cvec[j][6][2];
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
					tvec[0][0] = px + pos[i][0] - cvec[j][8][0];
					tvec[0][1] = py + pos[i][1] - cvec[j][8][1];
					tvec[0][2] = pz + pos[i][2] - cvec[j][8][2];
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
#ifdef debug
					if(i==1 && j==tris-1){
						ddum[0] = px+pos[i][0];
						ddum[1] = py+pos[i][1];
						ddum[2] = pz+pos[i][2];
						//cout << ddum[0] << " " << ddum[1] << " " << ddum[2] << endl;
						debug.push_back(ddum);
						debug2.push_back(vgrid);
					}
#endif
			}
		}
		}
		}
		vola[i] *= pow(dr/(double)gres,3);
	}
	vector <double> vol;
	for(unsigned int i=0; i<nvert; i++) vol.push_back(vola[i]);
	cout << " [OK]" << endl;

	//setting up fluid particles
	cout << "Defining fluid particles in a box ..." << endl;
	bool set = true;
	double fluid_vol = pow(dr,3);
	unsigned int nfluid = 0;
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
		for(int i=0; i<=floor((xmax+eps-xmin)/dr); i++){
		for(int j=0; j<=floor((ymax+eps-ymin)/dr); j++){
		for(int k=0; k<=floor((zmax+eps-zmin)/dr); k++){
			ddum[0] = xmin + (double)i*dr;
			ddum[1] = ymin + (double)j*dr;
			ddum[2] = zmin + (double)k*dr;
			pos.push_back(ddum);
			vol.push_back(fluid_vol);
			nfluid++;
		}}}
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
#ifndef debug
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
		buf[k].ep1 = nfluid+ep[i-nvert][0]; //AM-TODO: maybe + 1 as indices in fortran start with 1
		buf[k].ep2 = nfluid+ep[i-nvert][1];
		buf[k].ep3 = nfluid+ep[i-nvert][2];
		k++;
	}
#ifdef debug
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
	double time = 0.;
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

int hdf5_output (OutBuf *buf, int len, const char *filename, double *timevalue){
	hid_t		mem_type_id, loc_id, dataset_id, file_space_id, mem_space_id, xfer_plist_id;
	hsize_t	count[1], offset[1], dim[] = {len};
	herr_t	status;

	xfer_plist_id = H5Pcreate(H5P_FILE_ACCESS);
	loc_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, xfer_plist_id);
	H5Pclose(xfer_plist_id);
	file_space_id = H5Screate_simple(1, dim, NULL);
	mem_type_id = H5Tcreate(H5T_COMPOUND, sizeof(OutBuf));

	H5Tinsert(mem_type_id, "Coords_0"       , HOFFSET(OutBuf, x),       H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Coords_1"       , HOFFSET(OutBuf, y),       H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Coords_2"       , HOFFSET(OutBuf, z),       H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Normal_0"       , HOFFSET(OutBuf, nx),      H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Normal_1"       , HOFFSET(OutBuf, ny),      H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Normal_2"       , HOFFSET(OutBuf, nz),      H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Volume"         , HOFFSET(OutBuf, vol),     H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Surface"        , HOFFSET(OutBuf, surf),    H5T_NATIVE_DOUBLE);
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
