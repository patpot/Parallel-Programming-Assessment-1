//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	B[id] = A[id];
}

kernel void histogram(global const uchar* A, global int* B) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&B[bin_index]);
}

kernel void cumulative_histogram(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

kernel void histogram_lut(global int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = A[id] * (double)255 / A[255];
}

kernel void back_proj(global uchar* A, global int* LUT, global uchar* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]];
}