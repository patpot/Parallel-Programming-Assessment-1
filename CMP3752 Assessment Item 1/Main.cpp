#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void calculate_histogram() {

}

int main(int argc, char** argv) {
	// Handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	// Get user input for the bin number
	string userCommand;
	int bin_num = 0;
	std::cout << "Enter a bin number in range 0-256" << "\n";
	while (true) // while the user hasn't entered a valid number the program will keep running
	{
		getline(std::cin, userCommand); // Gets input from user
		if (userCommand == "") { std::cout << "Please enter a number." << "\n"; continue; } // Checks user input isn't empty

		try { bin_num = std::stoi(userCommand); } // Attempt to convert the user input to an integer
		catch (...) { std::cout << "Please enter an integer." << "\n"; continue; }

		if (bin_num >= 0 && bin_num <= 256) { break; } // Checks user input is in range
		else { std::cout << "Please enter a number in range 0-256." << "\n"; continue; }
	}


	//detect any potential exceptions
	try {
		// Read in the image from a given file path
		CImg<unsigned char> image_input(image_filename.c_str());
		// Display the image so we can see a before and after
		CImgDisplay disp_input(image_input, "input");

		// Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		// Create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		CImg<unsigned char> cb;
		CImg<unsigned char> cr;
		bool isColour = image_input.spectrum() == 3;
		if (isColour) {
			std::cout << "Image is RGB. " << std::endl;
			// Convert image to YCbCr so we have an intensity channel
			CImg<unsigned char> temp_image = image_input.get_RGBtoYCbCr();
			image_input = temp_image.get_channel(0);
			cb = temp_image.get_channel(1);
			cr = temp_image.get_channel(2);
		}
		else 
		{
			std::cout << "Image is Greyscale." << std::endl;
		}
		// Create a vector of size bin_num and initialize it with zeros
		std::vector<int> H(bin_num);
		// Calculate the size of the histogram buffer in bytes
		size_t hist_size = H.size() * sizeof(int);

		// Create device buffers for the input image, output image, histogram, cumulative histogram, and LUT
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		cl::Buffer dev_histogram_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer dev_cumulative_histogram_output(context, CL_MEM_READ_WRITE, hist_size);
		cl::Buffer dev_LUT_output(context, CL_MEM_READ_WRITE, hist_size);

		// Step 1: Calculate Histogram
		// Copy the input image to the device buffer for the input image
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		
		// Create kernel for histogram and set arguments
		cl::Kernel kernel_histogram = cl::Kernel(program, "histogram");
		kernel_histogram.setArg(0, dev_image_input);
		kernel_histogram.setArg(1, dev_histogram_output);

		// Ereate an event to measure the time for the histogram kernel execution
		cl::Event histogram_profiling_event;

		// Enqueue the histogram kernel to the command queue with a work size equal to the size of the input image
		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &histogram_profiling_event);
		// Read the histogram output from the device buffer to the host vector H
		queue.enqueueReadBuffer(dev_histogram_output, CL_TRUE, 0, hist_size, &H[0]);

		std::cout << "Histogram values:" << std::endl;
		for (int x : H) {
			std::cout << x << " ";
		}

		// Step 2: Calculate Cumulative Histogram
		// Create a vector of size bin_num to store the cumulative histogram
		std::vector<int> CH(bin_num);

		// Fill the device buffer for the cumulative histogram with zeros
		queue.enqueueFillBuffer(dev_cumulative_histogram_output, 0, 0, hist_size);

		// Create kernel for cumulative histogram and set arguments
		cl::Kernel kernel_histogram_cumulative = cl::Kernel(program, "cumulative_histogram");
		kernel_histogram_cumulative.setArg(0, dev_histogram_output);
		kernel_histogram_cumulative.setArg(1, dev_cumulative_histogram_output);

		// Create an event to measure the time for the cumulative histogram kernel execution
		cl::Event cumulative_histogram_profiling_event;

		// Enqueue the cumulative histogram kernel to the command queue with a work size equal to the size of the histogram vector
		queue.enqueueNDRangeKernel(kernel_histogram_cumulative, cl::NullRange, cl::NDRange(hist_size), cl::NullRange, NULL, &cumulative_histogram_profiling_event);
		// Read the cumulative histogram output from the device buffer to the host vector CH
		queue.enqueueReadBuffer(dev_cumulative_histogram_output, CL_TRUE, 0, hist_size, &CH[0]);

		std::cout << "Cumulative Histogram values:" << std::endl;
		for (int x : CH) {
			std::cout << x << " ";
		}

		// Step 3: Create Look-up Table of normalised values
		// Create a vector of size bin_num to store the LUT
		std::vector<int> LUT(bin_num);

		// Fill the device buffer for the LUT with zeros
		queue.enqueueFillBuffer(dev_LUT_output, 0, 0, hist_size);

		// Create kernel for LUT and set arguments
		cl::Kernel kernel_LUT = cl::Kernel(program, "histogram_lut");
		kernel_LUT.setArg(0, dev_cumulative_histogram_output);
		kernel_LUT.setArg(1, dev_LUT_output);

		// Create an event to measure the time for the LUT kernel execution
		cl::Event lut_profiling_event;

		// Enqueue the LUT kernel to the command queue with a work size equal to the size of the histogram vector
		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(hist_size), cl::NullRange, NULL, &lut_profiling_event);
		// Read the LUT output from the device buffer to the host vector LUT
		queue.enqueueReadBuffer(dev_LUT_output, CL_TRUE, 0, hist_size, &LUT[0]);


		// Step 4: Back-projection to map original intensities onto output image
		// Create kernel for Back projection and set arguments
		cl::Kernel kernel_back_proj = cl::Kernel(program, "back_proj");
		kernel_back_proj.setArg(0, dev_image_input);
		kernel_back_proj.setArg(1, dev_LUT_output);
		kernel_back_proj.setArg(2, dev_image_output);

		// Create an event to measure the time for the back projection kernel execution
		cl::Event back_proj_profiling_event;

		vector<unsigned char> output_buffer(image_input.size());
		// Enqueue the back projection kernel to the command queue with a work size equal to the size of the image input
		queue.enqueueNDRangeKernel(kernel_back_proj, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &back_proj_profiling_event);
		// Read the back projection output from the device buffer to the host vector output_buffer
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		// Output all execution times and memory transfers
		std::cout << std::endl;
		int hist_time = histogram_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogram_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "Histogram kernel execution time [ns]: " << hist_time << std::endl;
		std::cout << "Histogram memory transfer: " << GetFullProfilingInfo(histogram_profiling_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		int cumulative_hist_time = cumulative_histogram_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulative_histogram_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "Cumulative Histogram kernel execution time [ns]: " << cumulative_hist_time << std::endl;
		std::cout << "Cumulative Histogram memory transfer: " << GetFullProfilingInfo(cumulative_histogram_profiling_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		int lut_time = lut_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lut_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "LUT kernel execution time [ns]: " << lut_time << std::endl;
		std::cout << "LUT memory transfer: " << GetFullProfilingInfo(lut_profiling_event, ProfilingResolution::PROF_US) << std::endl << std::endl;;

		int back_proj_time = back_proj_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - back_proj_profiling_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		std::cout << "Back projection execution time [ns]: " << back_proj_time << std::endl;
		std::cout << "Back projection memory transfer: " << GetFullProfilingInfo(back_proj_profiling_event, ProfilingResolution::PROF_US) << std::endl;

		std::cout << "\nTotal program execution time [ns]: " << hist_time + cumulative_hist_time + lut_time + back_proj_time << std::endl;

		// "Reconstruct" our final output image from the histogram normalized values
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());

		if (isColour) {
			// The image was RGB, so we can't just take one channel from our output, we also need to add back in the CR and CB channels we cached previously.
			CImg<unsigned char> YCbCrImg(output_image.width(), output_image.height(), 1, 3);
			for (int x = 0; x < output_image.width(); x++) {
				for (int y = 0; y < output_image.height(); y++) {
					YCbCrImg(x, y, 0) = output_image(x, y);
					YCbCrImg(x, y, 1) = cb(x, y);
					YCbCrImg(x, y, 2) = cr(x, y);
				}
			}

			// Convert back to RGB for proper output
			output_image = YCbCrImg.get_YCbCrtoRGB();
		}

		CImgDisplay disp_output(output_image, "output");
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
