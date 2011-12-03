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

	cout << "Checking whether stl file is not ascii ...";
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
		cout << " [Failed]" << endl;
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
	cout << "Reading " << num_of_facets << " Facets ...";

	double dr = strtod(argv[2],NULL);

	// define variables
	vector< vector<double> > pos;
	vector< vector<double> > norm;
	vector< vector<double> >::iterator it;
	vector< vector<int> > ep;
	unsigned int nvert;
	vector<int> idum;
	vector<double> ddum;
	for (int i=0; i<3; i++) {
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
			stl_file.read((char *)&m_v_floats[i], sizeof(float));

		for (int i=0; i<3; i++) 
      ddum[i] = (double)m_v_floats[i];

		norm.push_back(ddum);

		for (int j=0;j<3;j++) {
			for (int i=0;i<3;i++) 
        ddum[i] = (double)m_v_floats[i+3*(j+1)];

			int k = 0;
			bool found = false;
			for (it = pos.begin(); it < pos.end(); it++) {
				double diff = 0;
				for (int i=0; i<3; i++) 
          diff += pow((*it)[i]-ddum[i],2);

				diff = sqrt(diff);
				if (diff < 1e-5*dr) {
					idum[j] = k;
					found = true;
					break;
				}
				k++;
			}
			if (!found) {
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
	if (num_of_facets != norm.size()) {
		cout << " [FAILED]" << endl;
		return READ_ERROR;
	}
	nvert = pos.size();
	cout << " [OK]" << endl;
	cout << "\n\tInformation:" << endl;
	cout << "\tOrigin of domain:           \t(" << xmin << ", " << ymin << ", " << zmin << ")\n";
	cout << "\tSize of domain:             \t(" << xmax-xmin << ", " << ymax-ymin << ", " << zmax-zmin << ")\n";
	cout << "\tNumber of vertices:         \t" << pos.size() << endl;
	cout << "\tNumber of boundary elements:\t" << norm.size() << "\n\n";

	//calculate surface and position of boundary elements
	cout << "Calculating surface and position of boundary elements ...";
	vector <double> surf;
	unsigned int nbe;
	vector <double> a;
	vector <double> b;
	vector <double> c;
	double xminp = 1e10, xminn = 1e10;
	double nminp = 0., nminn = 0.;
	for (unsigned int i=0; i<norm.size(); i++) {
		//formula: A = 1/4 sqrt(4*a^2*b^2-(a^2+b^2-c^2)^2)
		double a2 = 0.;
		double b2 = 0.;
		double c2 = 0.;
		ddum = vector <double> (3,0.);
		for (int j=0; j<3; j++) {
			ddum[j] += pos[ep[i][0]][j]/3.;
			ddum[j] += pos[ep[i][1]][j]/3.;
			ddum[j] += pos[ep[i][2]][j]/3.;
			a2 += pow(pos[ep[i][0]][j]-pos[ep[i][1]][j],2);
			b2 += pow(pos[ep[i][1]][j]-pos[ep[i][2]][j],2);
			c2 += pow(pos[ep[i][2]][j]-pos[ep[i][0]][j],2);
		}
		if (norm[i][2] > 1e-5 && xminp > ddum[2]) {
			xminp = ddum[2];
			nminp = norm[i][2];
		}
		if (norm[i][2] < -1e-5 && xminn > ddum[2]) {
			xminn = ddum[2];
			nminn = norm[i][2];
		}
		double A = 0.25*sqrt(4.*a2*b2-pow(a2+b2-c2,2));
		surf.push_back(A);
		pos.push_back(ddum);
		a.push_back(sqrt(a2));
		b.push_back(sqrt(b2));
		c.push_back(sqrt(c2));
	}
	nbe = pos.size() - nvert;
	cout << " [OK]" << endl;
	cout << "\n\tNormals information:" << endl;
	cout << "\tPositive (n.(0,0,1)) minimum z: " << xminp << " (" << nminp << ")\n";
	cout << "\tNegative (n.(0,0,1)) minimum z: " << xminn << " (" << nminn << ")\n\n";
	char cont= 'n';
	do {
		if(cont!='n') cout << "Wrong input. Answer with y or n." << endl;
		cout << "Swap normals (y/n): ";
		cin >> cont;
	} while (cont!='y' && cont!='n');
	if (cont=='y') {
		cout << "Swapping normals ...";
		for (unsigned int i=0; i<norm.size(); i++) {
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
	for (unsigned int idim=0; idim<3; idim++) {
		cont='n';
		do {
			if (cont!='n') cout << "Wrong input. Answer with y or n." << endl;
			if (idim==0) {
				cout << "X-periodicity (y/n): ";
			}
			else if (idim==1) {
				cout << "Y-periodicity (y/n): ";
			}
			else if (idim==2) {
				cout << "Z-periodicity (y/n): ";
			}
			cin >> cont;
		} while (cont!='y' && cont!='n');
		if (cont=='y') {
			per[idim] = true;
			cout << "Updating links ...";
			vector <int> newlink (nvert,-1);
			for (unsigned int i=0; i<nvert; i++) {
				if(fabs(pos[i][idim]-dmax[idim])<1e-5*dr){
					for(unsigned int j=0; j<nvert; j++){
						if(j==i) continue;
						if(sqrt(pow(pos[i][(idim+1)%3]-pos[j][(idim+1)%3],2.)+ \
						        pow(pos[i][(idim+2)%3]-pos[j][(idim+2)%3],2.)+ \
										pow(pos[j][idim]      -dmin[idim]        ,2.) ) < 1e-5*dr){
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
			for(unsigned int i=0; i<norm.size(); i++){
				for(unsigned int j=0; j<3; j++){
					if(newlink[ep[i][j]] != -1){ ep[i][j] = newlink[ep[i][j]]; }
				}
			}
			unsigned int onvert = nvert;
			for(int i=onvert-1; i>=0; i--){
				if(newlink[i] != -1){
					pos.erase(pos.begin()+i);
					nvert--;
					for(unsigned int j=0; j<norm.size(); j++){
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
	for(unsigned int i=0; i<norm.size(); i++){
		for(unsigned int j=0; j<3; j++){
			idum[0] = ep[i][(j+1)%3];
			idum[1] = ep[i][(j+2)%3];
			idum[2] = i;
			tri[ep[i][j]].push_back(idum);
		}
	}
	//sort neighbouring vertices
	//calculate volume (geometry factor)
	vector <double> vol (nvert,0.);
	const unsigned int gres = 60; // = dr/dr_grid
	const unsigned int gsize = gres*2+1; //makes sure that grid is large enough
	const double gdr = dr/(double)gres;
	float vgrid[gsize][gsize][gsize];
	double cvec[5][3], vnorm;
	double avnorm[3] = {0., 0., 0.};
	double eps=gdr*1e-4;
	vector <vector <double> > debug;
	vector <double> debug2;
	for(unsigned int i=0; i<nvert; i++){
		unsigned int tris = tri[i].size();
		for(unsigned int k=0; k<gsize; k++){ for(unsigned int l=0; l<gsize; l++){ for(unsigned int m=0; m<gsize; m++){
			vgrid[k][l][m] = 0.;
		}}}
		for(unsigned int j=0; j<3; j++) avnorm[j] = 0.;
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
			}
			//create cubes
			//setting up cube directions
			for(unsigned int n=0; n<3; n++) cvec[2][n] = norm[tri[i][j][2]][n]; //normal of boundary element
			vnorm = 0.;
			for(unsigned int n=0; n<3; n++){
				cvec[0][n] = pos[tri[i][j][0]][n]-pos[i][n]; //edge 1
				if(per[n]&&fabs(cvec[0][n])>2*dr)	cvec[0][n] += sgn(cvec[0][n])*(-dmax[n]+dmin[n]); //periodicty
				vnorm += pow(cvec[0][n],2);
			}
			vnorm = sqrt(vnorm);
			for(unsigned int n=0; n<3; n++) cvec[0][n] /= vnorm; 
			for(unsigned int n=0; n<3; n++)	cvec[1][n] = cvec[0][(n+1)%3]*cvec[2][(n+2)%3]-cvec[0][(n+2)%3]*cvec[2][(n+1)%3]; //cross product of normal and edge1
			vnorm = 0.;
			for(unsigned int n=0; n<3; n++){
				cvec[3][n] = pos[tri[i][j][1]][n]-pos[i][n]; //edge 2
				if(per[n]&&fabs(cvec[3][n])>2*dr)	cvec[3][n] += sgn(cvec[3][n])*(-dmax[n]+dmin[n]); //periodicty
				vnorm += pow(cvec[3][n],2);
				avnorm[n] -= norm[tri[i][j][2]][n];
			}
		  //if(i==130) cout << norm[tri[i][j][2]][0] << " " << norm[tri[i][j][2]][1] << " " << norm[tri[i][j][2]][2] << endl;
			vnorm = sqrt(vnorm);
			for(unsigned int n=0; n<3; n++) cvec[3][n] /= vnorm; 
			for(unsigned int n=0; n<3; n++)	cvec[4][n] = cvec[3][(n+1)%3]*cvec[2][(n+2)%3]-cvec[3][(n+2)%3]*cvec[2][(n+1)%3]; //cross product of normal and edge2
/*			if(i==0){
				for(unsigned int k=0; k<5; k++) cout << "cvec [" << j << "," << k << "]" << cvec[k][0] << " " << cvec[k][1] << " " << cvec[k][2] << endl;
			}*/
			//filling vgrid
			for(unsigned int k=0; k<gsize; k++){
			for(unsigned int l=0; l<gsize; l++){
			for(unsigned int m=0; m<gsize; m++){
				if(vgrid[k][l][m]>0.) continue; //grid point already full
				int ik = k-(gsize-1)/2;
				int il = l-(gsize-1)/2;
				int im = m-(gsize-1)/2;
				double px = ((double)ik)*gdr;
				double py = ((double)il)*gdr;
				double pz = ((double)im)*gdr;
				//if(i==0 && j == 0) cout << "now " << ik << " " << il << " " << im << " " << px << " " << py << " " << pz << endl;
				bool incube[5] = {false, false, false, false, false};
				for(unsigned int n=0; n<5; n++){
					double sp = px*cvec[n][0]+py*cvec[n][1]+pz*cvec[n][2];
					//if(i==0 && j==0) cout << sp << " " << px << " " << py << " " << pz << endl;
					//if(i==0 && j==0) cout << " 1 " << cvec[n][0]<< " " << cvec[n][1] << " " << cvec[n][2] << endl;
					if(fabs(sp)<=dr/2.+eps) incube[n] = true;
				}
				bool tmp = (vgrid[k][l][m]>0);
				if((incube[0] && incube[1] && incube[2]) || (incube[2] && incube[3] && incube[4])) vgrid[k][l][m] = 1.;
				/*if(i==0 && vgrid[k][l][m] && !tmp){
					ddum[0] = px+pos[i][0];
					ddum[1] = py+pos[i][1];
					ddum[2] = pz+pos[i][2];
					debug.push_back(ddum);
					//cout << "yeah: " << debug.size() << " " << pow(gsize,3) << endl;
				}*/
			}
			}
			}
			//end create cubes
		}
		if(tri[i][0][0] != tri[i][tris-1][1]){
			cout << " [FAILED]" << endl;
			return SORT_NOT_CLOSED;
		}
		//remove points based on planes (voronoi diagram & walls)
		for(unsigned int j=0; j<tris; j++){
			//set up plane normals and points
			double tvec[3][3];
			for(unsigned int n=0; n<3; n++){
				cvec[0][n] = pos[tri[i][j][0]][n]-pos[i][n]; //normal of plane voronoi
				if(per[n]&&fabs(cvec[0][n])>2*dr)	cvec[0][n] += sgn(cvec[0][n])*(-dmax[n]+dmin[n]); //periodicty
				cvec[1][n] = pos[i][n]+cvec[0][n]/2.; //position of plane voronoi
				tvec[0][n] = cvec[0][n]; // edge 1
				tvec[1][n] = pos[tri[i][j][1]][n]-pos[i][n]; // edge 2
				if(per[n]&&fabs(tvec[1][n])>2*dr)	tvec[1][n] += sgn(tvec[1][n])*(-dmax[n]+dmin[n]); //periodicty
				tvec[2][n] = avnorm[n]; // negative average normal
			}
			for(unsigned int n=0; n<3; n++){
				for(unsigned int k=0; k<3; k++){
					cvec[k+2][n] = tvec[k][(n+1)%3]*tvec[(k+1)%3][(n+2)%3]-tvec[k][(n+2)%3]*tvec[(k+1)%3][(n+1)%3]; //normals of tetrahedron planes
				}
			}
			double sp = 0.;
			for(unsigned int n=0; n<3; n++) sp += norm[tri[i][j][2]][n]*cvec[2][n]; //test whether normals point inward tetrahedron, if no flip normals
			if(sp > 0.){
				for(unsigned int k=0; k<3; k++){
					for(unsigned int n=0; n<3; n++)	cvec[k+2][n] *= -1.;
				}
			}

			//remove unwanted points and sum up for volume
			if(j==tris-1) vol[i] = 0.;
			for(unsigned int k=0; k<gsize; k++){
			for(unsigned int l=0; l<gsize; l++){
			for(unsigned int m=0; m<gsize; m++){
				if(vgrid[k][l][m]<eps) continue; //grid point already empty
				int ik = k-(gsize-1)/2;
				int il = l-(gsize-1)/2;
				int im = m-(gsize-1)/2;
				double px = pos[i][0] + ((double)ik)*gdr;
				double py = pos[i][1] + ((double)il)*gdr;
				double pz = pos[i][2] + ((double)im)*gdr;
				//voronoi plane
				tvec[0][0] = px - cvec[1][0];
				tvec[0][1] = py - cvec[1][1];
				tvec[0][2] = pz - cvec[1][2];
				double sp = 0.;
				for(unsigned int n=0; n<3; n++) sp += tvec[0][n]*cvec[0][n];
				if(sp>0.+eps){
					//bool tmp = (vgrid[k][l][m]>0);
					vgrid[k][l][m] = 0.;
					/*if(i==130 && tmp && vgrid[k][l][m]<eps){
						ddum[0] = px;
						ddum[1] = py;
						ddum[2] = pz;
						debug.push_back(ddum);
						debug2.push_back(2.);
					}*/
					continue;
				}
				else if(fabs(sp) < eps){
					vgrid[k][l][m] /= 2.;
				}
				//walls
				tvec[0][0] = px - pos[i][0];
				tvec[0][1] = py - pos[i][1];
				tvec[0][2] = pz - pos[i][2];
				bool half = false;
				for(unsigned int o=0; o<3; o++){
					sp = 0.;
					for(unsigned int n=0; n<3; n++) sp += tvec[0][n]*cvec[2+o][n];
					//if(sp < 0.+eps*(((double)(o==0))*2.-1.)) break;
					if(sp<0.-eps) break;
					if(fabs(sp)<eps && o==0) half=true;
					//bool tmp = (vgrid[k][l][m]>0);
					if(o==2 && !half){
						vgrid[k][l][m] = 0.;
					}
					else if(o==2 && half){
						vgrid[k][l][m] /= 2.;
					}
					/*if(i==130 && tmp && vgrid[k][l][m]<eps){
						ddum[0] = px;
						ddum[1] = py;
						ddum[2] = pz;
						debug.push_back(ddum);
						debug2.push_back(3.+j);
					}*/
				}
				if(j==tris-1)	vol[i] += (double)vgrid[k][l][m];
				/*if(j==tris-1 && i == 130 && vgrid[k][l][m] > 0){
					ddum[0] = px;
					ddum[1] = py;
					ddum[2] = pz;
					debug.push_back(ddum);
					debug2.push_back(vgrid[k][l][m]);
				}*/
			}
			}
			}
			vol[i] *= pow(dr/(double)gres,3);
		}
	}
	cout << " [OK]" << endl;

	//setting up fluid particles
	cout << "Defining fluid particles in a box ..." << endl;
	bool set = true;
	double fluid_vol = pow(dr,3);
	unsigned int nfluid = 0;
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
		for(int i=0; i<=floor((xmax-xmin)/dr); i++){
		for(int j=0; j<=floor((ymax-ymin)/dr); j++){
		for(int k=0; k<=floor((zmax-zmin)/dr); k++){
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

	//Prepare output structure
	cout << "Creating and initializing of output buffer ...";
	OutBuf *buf;
	buf = new OutBuf[pos.size()+debug.size()];
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
		buf[k].kpar = 2;
		buf[k].kfluid = 1;
		buf[k].kent = 1;
		buf[k].kparmob = 0;
		buf[k].iref = k;
		buf[k].ep1 = nfluid+ep[i-nvert][0]; //AM-TODO: maybe + 1 as indices in fortran start with 1
		buf[k].ep2 = nfluid+ep[i-nvert][1];
		buf[k].ep3 = nfluid+ep[i-nvert][2];
		k++;
	}
	for(unsigned int i=0; i<debug.size(); i++){
		buf[k].x = debug[i][0];
		buf[k].y = debug[i][1];
		buf[k].z = debug[i][2];
		pos.push_back(debug[i]);
		buf[k].nx = 0.;
		buf[k].ny = 0.;
		buf[k].nz = 0.;
		buf[k].vol = debug2[i];
		buf[k].surf = 0.;
		buf[k].kpar = 3;
		buf[k].kfluid = 1;
		buf[k].kent = 1;
		buf[k].kparmob = 0;
		buf[k].iref = 0;
		buf[k].ep1 = 0;
		buf[k].ep2 = 0;
		buf[k].ep3 = 0;
		k++;
	}
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
