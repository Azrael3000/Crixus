#include "crixus.h"
#include <hdf5.h>

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
