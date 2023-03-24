kernel void histogram(global const uchar* A, global int* B) {
	int id = get_global_id(0);
	atomic_inc(&B[A[id]]);
}

kernel void cumulative_histogram(global const int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0); // Get size of workload
	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]); // Summate all of the previous values to form a cumulative histogram
}

kernel void histogram_lut(global const int* A, global int* B) {
	int id = get_global_id(0);
	B[id] = A[id] * (double)255 / A[255]; // Normalize each value by 255 (colour depth in an 8 bit image)
}

kernel void back_proj(global const uchar* A, global const int* LUT, global uchar* B) {
	int id = get_global_id(0);
	B[id] = LUT[A[id]]; // Use our look up table to backwards project our original intensity to the new value
}