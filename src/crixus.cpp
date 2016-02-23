#include "crixus.h"
#include "return.h"
#include <hdf5.h>

using namespace std;

int main(int argc, char**argv){
  crixus_main(argc, argv);
  return 0;
}

int hdf5_output (OutBuf *buf, int len, const char *filename){
  hid_t    mem_type_id, loc_id, dataset_id, file_space_id, mem_space_id, xfer_plist_id;
  hsize_t  count[1], offset[1], dim[] = {len};
  herr_t  status;

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

int vtk_output (OutBuf *buf, int len, const char *filename){

  FILE *fid = fopen(filename, "w");

  if (fid == NULL) {
    cout << "Can't open file for writing: " << filename << "\n";
    return NO_WRITE_FILE;
  }

  // Header
  //====================================================================================
  fprintf(fid,"<?xml version='1.0'?>\n");
  fprintf(fid,"<VTKFile type= 'UnstructuredGrid'  version= '0.1'  byte_order= '%s'>\n",
    endianness[*(char*)&endian_int & 1]);
  fprintf(fid," <UnstructuredGrid>\n");
  fprintf(fid,"  <Piece NumberOfPoints='%d' NumberOfCells='%d'>\n", len, len);

  fprintf(fid,"   <PointData Scalars='Volume'>\n");

  size_t offset = 0;

  // Volume
  scalar_array(fid, "Float32", "Volume", offset);
  offset += sizeof(float)*len+sizeof(int);

  // Surface
  scalar_array(fid, "Float32", "Surface", offset);
  offset += sizeof(float)*len+sizeof(int);

  // particle type
  scalar_array(fid, "UInt32", "ParticleType", offset);
  offset += sizeof(uint)*len+sizeof(int);

  // fluid type
  scalar_array(fid, "UInt32", "FluidType", offset);
  offset += sizeof(uint)*len+sizeof(int);

  // kent
  scalar_array(fid, "UInt32", "KENT", offset);
  offset += sizeof(uint)*len+sizeof(int);

  // MovingBoundary
  scalar_array(fid, "UInt32", "MovingBoundary", offset);
  offset += sizeof(uint)*len+sizeof(int);

  // AbsoluteIndex
  scalar_array(fid, "UInt32", "AbsoluteIndex", offset);
  offset += sizeof(uint)*len+sizeof(int);

  // Normal
  vector_array(fid, "Float32", "Normal", 3, offset);
  offset += sizeof(float)*3*len+sizeof(int);

  // VertexParticle
  vector_array(fid, "UInt32", "VertexParticle", 3, offset);
  offset += sizeof(uint)*3*len+sizeof(int);

  fprintf(fid,"   </PointData>\n");

  // position
  fprintf(fid,"   <Points>\n");
  vector_array(fid, "Float64", 3, offset);
  offset += sizeof(double)*3*len+sizeof(int);
  fprintf(fid,"   </Points>\n");

  // Cells data
  fprintf(fid,"   <Cells>\n");
  scalar_array(fid, "Int32", "connectivity", offset);
  offset += sizeof(uint)*len+sizeof(int);
  scalar_array(fid, "Int32", "offsets", offset);
  offset += sizeof(uint)*len+sizeof(int);
  fprintf(fid,"  <DataArray type='Int32' Name='types' format='ascii'>\n");
  for (uint i = 0; i < len; i++)
    fprintf(fid,"%d\t", 1);
  fprintf(fid,"\n");
  fprintf(fid,"  </DataArray>\n");
  fprintf(fid,"   </Cells>\n");
  fprintf(fid,"  </Piece>\n");

  fprintf(fid," </UnstructuredGrid>\n");
  fprintf(fid," <AppendedData encoding='raw'>\n_");
  //====================================================================================

  // float entries
  int numbytes=sizeof(float)*len;

  // volume
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    float value = buf[i].vol;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // surface
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    float value = buf[i].surf;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // int entries
  numbytes=sizeof(uint)*len;

  // ParticleType
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = buf[i].kpar;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // FluidType
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = buf[i].kfluid;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // KENT
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = buf[i].kent;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // MovingBoundary
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = buf[i].kparmob;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // AbsoluteIndex
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = buf[i].iref;
    fwrite(&value, sizeof(value), 1, fid);
  }

  // float vector entries
  numbytes=sizeof(float)*3*len;

  // normal
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    float value[3];
    value[0] = buf[i].nx;
    value[1] = buf[i].ny;
    value[2] = buf[i].nz;
    fwrite(value, sizeof(value[0]), 3, fid);
  }

  // int vector entries
  numbytes=sizeof(uint)*3*len;

  // vertexParticles
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value[3];
    value[0] = buf[i].ep1;
    value[1] = buf[i].ep2;
    value[2] = buf[i].ep3;
    fwrite(value, sizeof(value[0]), 3, fid);
  }

  // double vector entries
  numbytes=sizeof(double)*3*len;

  // position
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    double value[3];
    value[0] = (double) buf[i].x;
    value[1] = (double) buf[i].y;
    value[2] = (double) buf[i].z;
    fwrite(value, sizeof(value[0]), 3, fid);
  }

  numbytes=sizeof(uint)*len;
  // connectivity
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = i;
    fwrite(&value, sizeof(value), 1, fid);
  }
  // offsets
  fwrite(&numbytes, sizeof(numbytes), 1, fid);
  for (uint i=0; i < len; i++) {
    uint value = i+1;
    fwrite(&value, sizeof(value), 1, fid);
  }

  fprintf(fid," </AppendedData>\n");
  fprintf(fid,"</VTKFile>");

  fclose(fid);

  return 0;
}

// Wrapper for {hdf5,vtk}_output()
int generic_output(OutBuf *buf, int start, int nelem, const char* outname_c, int opt)
{
  string outname(outname_c);
  int err = 0;

  if(opt==2){
    outname = "0." + outname + ".h5sph";
    cout << "Writing output to file " << outname << " ...";
    fflush(stdout);
    err = hdf5_output( buf + start, nelem, outname.c_str());
  }
  else if(opt==1){
    outname += ".vtu";
    cout << "Writing output to file " << outname << " ...";
    fflush(stdout);
    err = vtk_output( buf + start, nelem, outname.c_str());
  }
  else
    // something's really wrong if opt is not 1 nor 2
    return err = INTERNAL_ERROR;

  if(err==0){ cout << " [OK]" << endl; }
  else {
    cout << " [FAILED]" << endl;
    err = WRITE_FAIL;
  }
  return err;
}

/* auxiliary functions to write data array entrypoints */
inline void
scalar_array(FILE *fid, const char *type, const char *name, size_t offset)
{
  fprintf(fid, "  <DataArray type='%s' Name='%s' "
      "format='appended' offset='%zu'/>\n",
      type, name, offset);
}

inline void
vector_array(FILE *fid, const char *type, const char *name, uint dim, size_t offset)
{
  fprintf(fid, "  <DataArray type='%s' Name='%s' NumberOfComponents='%u' "
      "format='appended' offset='%zu'/>\n",
      type, name, dim, offset);
}

inline void
vector_array(FILE *fid, const char *type, uint dim, size_t offset)
{
  fprintf(fid, "  <DataArray type='%s' NumberOfComponents='%u' "
      "format='appended' offset='%zu'/>\n",
      type, dim, offset);
}

/* Function for vertex reduction of stl file */
void find_in_cell(list<intb>::iterator il_neib, list<intb>::iterator il_end, const float xi, const float yi, const float zi, vector<uint>& new_ids_map, vector<vector<float> >& all_verts, const float dr, const uint new_id)
{
  for(; il_neib != il_end; il_neib++) {
    const float xij = all_verts[il_neib->idx][0] - xi;
    const float yij = all_verts[il_neib->idx][1] - yi;
    const float zij = all_verts[il_neib->idx][2] - zi;
    const float dist = xij*xij + yij*yij + zij*zij;
    if (dist < 1e-10*dr*dr) {
      il_neib->found = true;
      new_ids_map[il_neib->idx] = new_id;
    }
  }
}

void offset_find_in_cell(int x, int y, int z, mint3 key, map<mint3, list<intb> >cell_map, list<intb>::iterator il_neib, list<intb>::iterator il_end, const float xi, const float yi, const float zi, vector<uint>& new_ids_map, vector<vector<float> >& all_verts, const float dr, const uint new_id)
{
  map<mint3, list<intb> >::iterator search;
  mint3 neib_cell = mint3(key.x+x, key.y+y, key.z+z);
  search = cell_map.find(neib_cell);
  if(search != cell_map.end()) {
    list<intb>::iterator il_neib = search->second.begin();
    list<intb>::iterator il_end = search->second.end();
    find_in_cell(il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
  }
}

void remove_duplicate_vertices(vector<vector<float> >& all_verts, vector<vector<uint> >& old_ids, vector<vector<float> >& new_verts, vector<vector<uint> >& new_ids, const float dr)
{
  vector< vector <uint> >::iterator it;
  // construct map
  map<mint3, list<intb> > cell_map;
  map<mint3, list<intb> >::iterator search;
  for(it = old_ids.begin(); it < old_ids.end(); it++) {
    for(uint i = 0; i<3; i++) {
      float x = all_verts[(*it)[i]][0];
      float y = all_verts[(*it)[i]][1];
      float z = all_verts[(*it)[i]][2];
      mint3 cell((int)(x/dr), (int)(y/dr), (int)(z/dr));
      intb ib((*it)[i], false);
      // find if key exists
      search = cell_map.find(cell);
      // if it does then append to list
      if (search != cell_map.end())
        search->second.push_back(ib);
      // if not create a new map
      else
        cell_map[cell].push_back(ib);
    }
  }
  // map output
  map<mint3, list<intb> >::iterator im;
  vector<uint> new_ids_map;
  new_ids_map.resize(all_verts.size());
  // identify similar vertices
  for(im = cell_map.begin(); im != cell_map.end(); im++) {
    list<intb>::iterator il;
    for(il = im->second.begin(); il != im->second.end(); il++) {
      if(il->found)
        continue;
      il->found = true; // a bit unnecessary probably
      new_verts.push_back(all_verts[il->idx]);
      const uint new_id = new_verts.size()-1;
      new_ids_map[il->idx] = new_id;
      // vert pos
      const float xi = all_verts[il->idx][0];
      const float yi = all_verts[il->idx][1];
      const float zi = all_verts[il->idx][2];
      // look in same and neighbouring cells for potentially identical vertices
      mint3 neib_cell;
      // x+=0 y+=0 z+=0
      list<intb>::iterator il_neib;
      list<intb>::iterator il_end;
      il_neib = il;
      il_neib++;
      il_end = im->second.end();
      find_in_cell(il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=0 y+=0 z+=1
      offset_find_in_cell( 0, 0, 1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=0 y+=1 z+=-1
      offset_find_in_cell( 0, 1,-1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=0 y+=1 z+=0
      offset_find_in_cell( 0, 1, 0, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=0 y+=1 z+=1
      offset_find_in_cell( 0, 1, 1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=-1 z+=-1
      offset_find_in_cell( 1,-1,-1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=-1 z+=0
      offset_find_in_cell( 1,-1, 0, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=-1 z+=1
      offset_find_in_cell( 1,-1, 1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=0 z+=-1
      offset_find_in_cell( 1, 0,-1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=0 z+=0
      offset_find_in_cell( 1, 0, 0, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=0 z+=1
      offset_find_in_cell( 1, 0, 1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=1 z+=-1
      offset_find_in_cell( 1, 1,-1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=1 z+=0
      offset_find_in_cell( 1, 1, 0, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
      // x+=1 y+=1 z+=1
      offset_find_in_cell( 1, 1, 1, im->first, cell_map, il_neib, il_end, xi, yi, zi, new_ids_map, all_verts, dr, new_id);
    }
  }
  // write new ids into vector
  vector<uint> idum;
  idum.resize(3,0);
  for (it = old_ids.begin(); it != old_ids.end(); it++) {
    idum[0] = new_ids_map[(*it)[0]];
    idum[1] = new_ids_map[(*it)[1]];
    idum[2] = new_ids_map[(*it)[2]];
    new_ids.push_back(idum);
  }
  return;
}
