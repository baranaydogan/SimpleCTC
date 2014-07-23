#include "BoundaryDetection.h"

void MarkRegions(int8_t* out, uint8_t* inp, int32_t m, int32_t n) {

	// Inside the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++) {

		for (int32_t y=1; y<(n-1); y++)	{

			if ( inp[x+y*m] && inp[(x-1)+y*m] && inp[(x+1)+y*m] && inp[x+(y-1)*m] && inp[x+(y+1)*m] )
				out[x+y*m] = 1;
			else if (inp[x+y*m])
				out[x+y*m] = 0;
			else
				out[x+y*m] = -1;
		}
	}

	// Top boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		if ( inp[y*m] && inp[(y-1)*m] && inp[(y+1)*m] && inp[1+y*m])
			out[y*m] = 1;
		else if (inp[y*m])
			out[y*m] = 0;
		else
			out[y*m] = -1;
	}

	// Bottom boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		if ( inp[(m-1)+y*m] && inp[(m-1)+(y-1)*m] && inp[(m-1)+(y+1)*m] && inp[(m-2)+y*m])
			out[(m-1)+y*m] = 1;
		else if (inp[(m-1)+y*m])
			out[(m-1)+y*m] = 0;
		else
			out[(m-1)+y*m] = -1;
	}

	// Left boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		if ( inp[x] && inp[(x-1)] && inp[(x+1)] && inp[x+m])
			out[x] = 1;
		else if (inp[x])
			out[x] = 0;
		else
			out[x] = -1;
	}

	// Right boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		if ( inp[x+(n-1)*m] && inp[(x-1)+(n-1)*m] && inp[(x+1)+(n-1)*m] && inp[x+(n-2)*m])
			out[x+(n-1)*m] = 1;
		else if (inp[x+(n-1)*m])
			out[x+(n-1)*m] = 0;
		else
			out[x+(n-1)*m] = -1;
	}

	// Top-Left corner
	if ( inp[0] && inp[1] && inp[m] )
		out[0] = 1;
	else if (inp[0])
		out[0] = 0;
	else
		out[0] = 0-1;

	// Bottom-Left corner
	if ( inp[m-1] && inp[m-2] && inp[(m-1)+m] )
		out[m-1] = 1;
	else if (inp[m-1])
		out[m-1] = 0;
	else
		out[m-1] = -1;

	// Top-Right corner
	if ( inp[(n-1)*m] && inp[(n-2)*m] && inp[1+(n-1)*m] )
		out[(n-1)*m] = 1;
	else if (inp[(n-1)*m])
		out[(n-1)*m] = 0;
	else
		out[(n-1)*m] = -1;

	// Bottom-Right corner
	if ( inp[(m-1)+(n-1)*m] && inp[(m-1)+(n-2)*m] && inp[(m-2)+(n-1)*m] )
		out[(m-1)+(n-1)*m] = 1;
	else if (inp[(m-1)+(n-1)*m])
		out[(m-1)+(n-1)*m] = 0;
	else
		out[(m-1)+(n-1)*m] = -1;
}

void MarkRegions(int8_t* out, uint8_t* inp, int32_t m, int32_t n, int32_t p) {

	// Inside the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++) {
		for (int32_t y=1; y<(n-1); y++)	{
			for (int32_t z=1; z<(p-1); z++)	{

				if ( inp[x+y*m+z*m*n] &&
						inp[(x-1)+y*m+z*m*n] &&
						inp[(x+1)+y*m+z*m*n] &&
						inp[x+(y-1)*m+z*m*n] &&
						inp[x+(y+1)*m+z*m*n] &&
						inp[x+y*m+(z-1)*m*n] &&
						inp[x+y*m+(z+1)*m*n]
				)
					out[x+y*m+z*m*n] = 1;
				else if (inp[x+y*m+z*m*n])
					out[x+y*m+z*m*n] = 0;
				else
					out[x+y*m+z*m*n] = -1;

			}
		}
	}

	// Top boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		for (int32_t z=1; z<(p-1); z++)	{
			if ( inp[y*m+z*m*n] &&
					inp[(y-1)*m+z*m*n] &&
					inp[(y+1)*m+z*m*n] &&
					inp[y*m+(z-1)*m*n] &&
					inp[y*m+(z+1)*m*n] &&
					inp[1+y*m+z*m*n]
			)
				out[y*m+z*m*n] = 1;
			else if (inp[y*m+z*m*n])
				out[y*m+z*m*n] = 0;
			else
				out[y*m+z*m*n] = -1;
		}
	}

	// Bottom boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		for (int32_t z=1; z<(p-1); z++)	{
			if ( inp[(m-1)+y*m+z*m*n] &&
					inp[(m-1)+(y-1)*m+z*m*n] &&
					inp[(m-1)+(y+1)*m+z*m*n] &&
					inp[(m-1)+y*m+(z-1)*m*n] &&
					inp[(m-1)+y*m+(z+1)*m*n] &&
					inp[(m-2)+y*m+z*m*n]
			)
				out[(m-1)+y*m+z*m*n] = 1;
			else if (inp[(m-1)+y*m+z*m*n])
				out[(m-1)+y*m+z*m*n] = 0;
			else
				out[(m-1)+y*m+z*m*n] = -1;
		}
	}

	// Left boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		for (int32_t z=1; z<(p-1); z++)	{
			if ( inp[x+z*m*n] &&
					inp[(x-1)+z*m*n] &&
					inp[(x+1)+z*m*n] &&
					inp[x+(z-1)*m*n] &&
					inp[x+(z+1)*m*n] &&
					inp[x+m+z*m*n]
			)
				out[x+z*m*n] = 1;
			else if (inp[x+z*m*n])
				out[x+z*m*n] = 0;
			else
				out[x+z*m*n] = -1;
		}
	}

	// Right boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		for (int32_t z=1; z<(p-1); z++)	{
			if ( inp[x+(n-1)*m+z*m*n] &&
					inp[(x-1)+(n-1)*m+z*m*n] &&
					inp[(x+1)+(n-1)*m+z*m*n] &&
					inp[x+(n-1)*m+(z-1)*m*n] &&
					inp[x+(n-1)*m+(z+1)*m*n] &&
					inp[x+(n-2)*m+z*m*n]
			)
				out[x+(n-1)*m+z*m*n] = 1;
			else if (inp[x+(n-1)*m+z*m*n])
				out[x+(n-1)*m+z*m*n] = 0;
			else
				out[x+(n-1)*m+z*m*n] = -1;
		}
	}

	// Front boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		for (int32_t y=1; y<(n-1); y++)	{
			if ( inp[x+y*m] &&
					inp[x-1+y*m] &&
					inp[x+1+y*m] &&
					inp[x+(y-1)*m] &&
					inp[x+(y+1)*m] &&
					inp[x+y*m+m*n]
			)
				out[x+y*m] = 1;
			else if (inp[x+y*m])
				out[x+y*m] = 0;
			else
				out[x+y*m] = -1;
		}
	}

	// Back boundary of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		for (int32_t y=1; y<(n-1); y++)	{
			if ( inp[x+y*m+(p-1)*m*n] &&
					inp[x-1+y*m+(p-1)*m*n] &&
					inp[x+1+y*m+(p-1)*m*n] &&
					inp[x+(y-1)*m+(p-1)*m*n] &&
					inp[x+(y+1)*m+(p-1)*m*n] &&
					inp[x+y*m+(p-2)*m*n]
			)
				out[x+y*m+(p-1)*m*n] = 1;
			else if (inp[x+y*m+(p-1)*m*n])
				out[x+y*m+(p-1)*m*n] = 0;
			else
				out[x+y*m+(p-1)*m*n] = -1;
		}
	}


	// Front-top boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		if ( inp[y*m] &&
				inp[(y-1)*m] &&
				inp[(y+1)*m] &&
				inp[1+y*m] &&
				inp[y*m+m*n]
		)
			out[y*m] = 1;
		else if (inp[y*m])
			out[y*m] = 0;
		else
			out[y*m] = -1;
	}

	// Front-bottom boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		if ( inp[(m-1)+y*m] &&
				inp[(m-1)+(y-1)*m] &&
				inp[(m-1)+(y+1)*m] &&
				inp[(m-2)+y*m] &&
				inp[(m-1)+y*m+m*n]
		)
			out[(m-1)+y*m] = 1;
		else if (inp[(m-1)+y*m])
			out[(m-1)+y*m] = 0;
		else
			out[(m-1)+y*m] = -1;
	}

	// Front-left boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		if ( inp[x] &&
				inp[x-1] &&
				inp[x+1] &&
				inp[x+m] &&
				inp[x+m*n]
		)
			out[x] = 1;
		else if (inp[x])
			out[x] = 0;
		else
			out[x] = -1;
	}

	// Front-right boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		if ( inp[x+(n-1)*m] &&
				inp[x-1+(n-1)*m] &&
				inp[x+1+(n-1)*m] &&
				inp[x+(n-2)*m] &&
				inp[x+(n-1)*m+m*n]
		)
			out[x+(n-1)*m] = 1;
		else if (inp[x+(n-1)*m])
			out[x+(n-1)*m] = 0;
		else
			out[x+(n-1)*m] = -1;
	}

	// Back-top boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		if ( inp[y*m+(p-1)*m*n] &&
				inp[(y-1)*m+(p-1)*m*n] &&
				inp[(y+1)*m+(p-1)*m*n] &&
				inp[1+y*m+(p-1)*m*n] &&
				inp[y*m+(p-2)*m*n]
		)
			out[y*m+(p-1)*m*n] = 1;
		else if (inp[y*m+(p-1)*m*n])
			out[y*m+(p-1)*m*n] = 0;
		else
			out[y*m+(p-1)*m*n] = -1;
	}

	// Back-bottom boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t y=1; y<(n-1); y++)	{
		if ( inp[(m-1)+y*m+(p-1)*m*n] &&
				inp[(m-1)+(y-1)*m+(p-1)*m*n] &&
				inp[(m-1)+(y+1)*m+(p-1)*m*n] &&
				inp[(m-2)+y*m+(p-1)*m*n] &&
				inp[(m-1)+y*m+(p-2)*m*n]
		)
			out[(m-1)+y*m+(p-1)*m*n] = 1;
		else if (inp[(m-1)+y*m+(p-1)*m*n])
			out[(m-1)+y*m+(p-1)*m*n] = 0;
		else
			out[(m-1)+y*m+(p-1)*m*n] = -1;
	}

	// Back-left boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		if ( inp[x+(p-1)*m*n] &&
				inp[x-1+(p-1)*m*n] &&
				inp[x+1+(p-1)*m*n] &&
				inp[x+m+(p-1)*m*n] &&
				inp[x+(p-2)*m*n]
		)
			out[x+(p-1)*m*n] = 1;
		else if (inp[x+(p-1)*m*n])
			out[x+(p-1)*m*n] = 0;
		else
			out[x+(p-1)*m*n] = -1;
	}

	// Back-right boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t x=1; x<(m-1); x++)	{
		if ( inp[x+(n-1)*m+(p-1)*m*n] &&
				inp[x-1+(n-1)*m+(p-1)*m*n] &&
				inp[x+1+(n-1)*m+(p-1)*m*n] &&
				inp[x+(n-2)*m+(p-1)*m*n] &&
				inp[x+(n-1)*m+(p-2)*m*n]
		)
			out[x+(n-1)*m+(p-1)*m*n] = 1;
		else if (inp[x+(n-1)*m+(p-1)*m*n])
			out[x+(n-1)*m+(p-1)*m*n] = 0;
		else
			out[x+(n-1)*m+(p-1)*m*n] = -1;
	}


	// Left-top boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t z=1; z<(p-1); z++)	{
		if ( inp[z*m*n] &&
				inp[(z-1)*m*n] &&
				inp[(z+1)*m*n] &&
				inp[1+z*m*n] &&
				inp[m+z*m*n]
		)
			out[z*m*n] = 1;
		else if (inp[z*m*n])
			out[z*m*n] = 0;
		else
			out[z*m*n] = -1;
	}

	// Left-bottom boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t z=1; z<(p-1); z++)	{
		if ( inp[(m-1)+z*m*n] &&
				inp[(m-1)+(z-1)*m*n] &&
				inp[(m-1)+(z+1)*m*n] &&
				inp[(m-2)+z*m*n] &&
				inp[(m-1)+m+z*m*n]
		)
			out[(m-1)+z*m*n] = 1;
		else if (inp[(m-1)+z*m*n])
			out[(m-1)+z*m*n] = 0;
		else
			out[(m-1)+z*m*n] = -1;
	}

	// Right-top boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t z=1; z<(p-1); z++)	{
		if ( inp[(n-1)*m+z*m*n] &&
				inp[(n-1)*m+(z-1)*m*n] &&
				inp[(n-1)*m+(z+1)*m*n] &&
				inp[(n-2)*m+z*m*n] &&
				inp[1+(n-1)*m+z*m*n]
		)
			out[(n-1)*m+z*m*n] = 1;
		else if (inp[(n-1)*m+z*m*n])
			out[(n-1)*m+z*m*n] = 0;
		else
			out[(n-1)*m+z*m*n] = -1;
	}

	// Right-bottom boundary line of the image
	#pragma omp parallel for shared(inp,out) schedule(dynamic)
	for (int32_t z=1; z<(p-1); z++)	{
		if ( inp[(m-1)+(n-1)*m+z*m*n] &&
				inp[(m-1)+(n-1)*m+(z-1)*m*n] &&
				inp[(m-1)+(n-1)*m+(z+1)*m*n] &&
				inp[(m-2)+(n-1)*m+z*m*n] &&
				inp[(m-1)+(n-2)*m+z*m*n]
		)
			out[(m-1)+(n-1)*m+z*m*n] = 1;
		else if (inp[(m-1)+(n-1)*m+z*m*n])
			out[(m-1)+(n-1)*m+z*m*n] = 0;
		else
			out[(m-1)+(n-1)*m+z*m*n] = -1;
	}



	// Front-top-Left corner
	if ( inp[0] &&
			inp[1] &&
			inp[m] &&
			inp[m*n]
	)
		out[0] = 1;
	else if (inp[0])
		out[0] = 0;
	else
		out[0] = 0-1;

	// Front-bottom-Left corner
	if ( inp[m-1] &&
			inp[m-2] &&
			inp[(m-1)+m] &&
			inp[(m-1)+m*n]
	)
		out[m-1] = 1;
	else if (inp[m-1])
		out[m-1] = 0;
	else
		out[m-1] = -1;

	// Front-top-Right corner
	if ( inp[(n-1)*m] &&
			inp[(n-2)*m] &&
			inp[1+(n-1)*m] &&
			inp[(n-1)*m+m*n]
	)
		out[(n-1)*m] = 1;
	else if (inp[(n-1)*m])
		out[(n-1)*m] = 0;
	else
		out[(n-1)*m] = -1;

	// Front-bottom-Right corner
	if ( inp[(m-1)+(n-1)*m] &&
			inp[(m-1)+(n-2)*m] &&
			inp[(m-2)+(n-1)*m] &&
			inp[(m-1)+(n-1)*m+m*n]
	)
		out[(m-1)+(n-1)*m] = 1;
	else if (inp[(m-1)+(n-1)*m])
		out[(m-1)+(n-1)*m] = 0;
	else
		out[(m-1)+(n-1)*m] = -1;

	//  Back-top-Left corner
	if ( inp[(p-1)*m*n] &&
			inp[1+(p-1)*m*n] &&
			inp[m+(p-1)*m*n] &&
			inp[(p-2)*m*n]
	)
		out[(p-1)*m*n] = 1;
	else if (inp[(p-1)*m*n])
		out[(p-1)*m*n] = 0;
	else
		out[(p-1)*m*n] = 0-1;

	// Back-bottom-Left corner
	if ( inp[m-1+(p-1)*m*n] &&
			inp[m-2+(p-1)*m*n] &&
			inp[(m-1)+m+(p-1)*m*n] &&
			inp[(m-1)+(p-2)*m*n]
	)
		out[m-1+(p-1)*m*n] = 1;
	else if (inp[m-1+(p-1)*m*n])
		out[m-1+(p-1)*m*n] = 0;
	else
		out[m-1+(p-1)*m*n] = -1;

	// Back-top-Right corner
	if ( inp[(n-1)*m+(p-1)*m*n] &&
			inp[(n-2)*m+(p-1)*m*n] &&
			inp[1+(n-1)*m+(p-1)*m*n] &&
			inp[(n-1)*m+(p-2)*m*n]
	)
		out[(n-1)*m+(p-1)*m*n] = 1;
	else if (inp[(n-1)*m+(p-1)*m*n])
		out[(n-1)*m+(p-1)*m*n] = 0;
	else
		out[(n-1)*m+(p-1)*m*n] = -1;

	// Back-bottom-Right corner
	if ( inp[(m-1)+(n-1)*m+(p-1)*m*n] &&
			inp[(m-1)+(n-2)*m+(p-1)*m*n] &&
			inp[(m-2)+(n-1)*m+(p-1)*m*n] &&
			inp[(m-1)+(n-1)*m+(p-2)*m*n]
	)
		out[(m-1)+(n-1)*m+(p-1)*m*n] = 1;
	else if (inp[(m-1)+(n-1)*m+(p-1)*m*n])
		out[(m-1)+(n-1)*m+(p-1)*m*n] = 0;
	else
		out[(m-1)+(n-1)*m+(p-1)*m*n] = -1;


}
