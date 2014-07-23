#include "DistanceTransform.h"

int32_t F(int32_t x, int32_t i, int32_t gi) {
	return (x-i)*(x-i) + gi;
}

int32_t Sep(int32_t i, int32_t u, int32_t gi, int32_t gu) {
	return floor( (u*u - i*i + gu - gi) / (2*(u-i)) );
}

void DistanceTransform(double* dt, uint8_t* b, int32_t m, int32_t n) {

	int32_t inf 				= m+n;

	int32_t* g;
	g							=	new int32_t[m*n];

#pragma omp parallel for shared(b,g) schedule(dynamic)
	for (int32_t x=0; x<m; x++) {

		if (b[x])
			g[x] = 0;
		else
			g[x] = inf;

		// Scan 1
		for (int32_t y=1; y<n; y++)	{
			if (b[x+y*m])
				g[x+y*m] = 0;
			else
				g[x+y*m] = 1 + g[x+(y-1)*m];
		}

		// Scan 2
		for (int32_t y=(n-2); y>= 0; y--)	{
			if (g[x+(y+1)*m] < g[x+y*m])
				g[x+y*m] = 1 + g[x+(y+1)*m];
		}

	}

	// Second phase
	int32_t* s;
	int32_t* t;

	int32_t q = 0;
	int32_t w;

#pragma omp parallel for shared(g,dt) private(q,w,s,t) schedule(dynamic)
	for (int32_t y=0; y<n; y++)	{

		s		=	new int32_t[m];
		t		=	new int32_t[m];

		q 		= 0;
		s[0] 	= 0;
		t[0] 	= 0;

		// Scan 3
		for (int32_t x=1; x<m; x++)	{
			while (q >= 0 && ( F(t[q],s[q],g[s[q]+y*m]*g[s[q]+y*m]) > F(t[q],x,g[x+y*m]*g[x+y*m]) ) )
				q--;
			if (q < 0)	{
				q = 0;
				s[0] = x;
			}	else	{
				w = 1 + Sep(s[q],x,g[s[q]+y*m]*g[s[q]+y*m],g[x+y*m]*g[x+y*m]);
				if (w < m)	{
					q++;
					s[q] = x;
					t[q] = w;
				}
			}
		}

		// Scan 4
		for (int32_t x=(m-1); x>=0; x--)	{
			dt[x+y*m] = sqrt(F(x,s[q],g[s[q]+y*m]*g[s[q]+y*m]));
			if (x == t[q])
				q--;
		}

		delete[] s;
		delete[] t;

	}

	delete[] g;

}

void DistanceTransform(double* dtz, uint8_t* b, int32_t m, int32_t n, int32_t p) {

	int32_t inf 				= m+n+p;

	int32_t* g;

	int32_t* dt;
	dt							=	new int32_t[m*n*p];

	int32_t* s;
	int32_t* t;

	int32_t q = 0;
	int32_t w;

#pragma omp parallel for shared(b,dt) private(g,q,w,s,t) schedule(dynamic)
	for (int32_t z=0; z<p; z++) {

		g	=	new int32_t[m*n];
		s	=	new int32_t[m];
		t	=	new int32_t[m];

		for (int32_t x=0; x<m; x++) {

			if (b[x+z*m*n])
				g[x] = 0;
			else
				g[x] = inf;

			// Scan 1
			for (int32_t y=1; y<n; y++)	{
				if (b[x+y*m+z*m*n])
					g[x+y*m] = 0;
				else
					g[x+y*m] = 1 + g[x+(y-1)*m];
			}

			// Scan 2
			for (int32_t y=(n-2); y>= 0; y--)	{
				if (g[x+(y+1)*m] < g[x+y*m])
					g[x+y*m] = 1 + g[x+(y+1)*m];
			}

		}

		// Second phase
		for (int32_t y=0; y<n; y++)	{

			q 		= 0;
			s[0] 	= 0;
			t[0] 	= 0;

			// Scan 3
			for (int32_t x=1; x<m; x++)	{
				while (q >= 0 && ( F(t[q],s[q],g[s[q]+y*m]*g[s[q]+y*m]) > F(t[q],x,g[x+y*m]*g[x+y*m])) )
					q--;
				if (q < 0)	{
					q = 0;
					s[0] = x;
				}	else	{
					w = 1 + Sep(s[q],x,g[s[q]+y*m]*g[s[q]+y*m],g[x+y*m]*g[x+y*m]);
					if (w < m)	{
						q++;
						s[q] = x;
						t[q] = w;
					}
				}
			}

			// Scan 4
			for (int32_t x=(m-1); x>=0; x--)	{
				dt[x+y*m+z*m*n] = F(x,s[q],g[s[q]+y*m]*g[s[q]+y*m]);
				if (x == t[q])
					q--;
			}

		}

		delete[] s;
		delete[] t;
		delete[] g;


	}

	int32_t* sz;
	int32_t* tz;

#pragma omp parallel for shared(dt,dtz) private(q,w,sz,tz) schedule(dynamic)
	for (int32_t x=0; x<m; x++) {

		sz		=	new int32_t[p];
		tz		=	new int32_t[p];

		for (int32_t y=0; y<n; y++)	{

			q 		= 0;
			sz[0]	= 0;
			tz[0] 	= 0;

			// Scan 5
			for (int32_t z=1; z<p; z++)	{
				while (q >= 0 && ( F(tz[q],sz[q],dt[x+y*m+sz[q]*m*n]) > F(tz[q],z,dt[x+y*m+z*m*n]) ) )
					q--;
				if (q < 0)	{
					q = 0;
					sz[0] = z;
				}	else	{
					w = 1 + Sep(sz[q],z,dt[x+y*m+sz[q]*m*n],dt[x+y*m+z*m*n]);
					if (w < p)	{
						q++;
						sz[q] = z;
						tz[q] = w;
					}
				}
			}

			// Scan 6
			for (int32_t z=(p-1); z>=0; z--)	{
				dtz[x+y*m+z*m*n] = sqrt(F(z,sz[q],dt[x+y*m+sz[q]*m*n]));
				if (z == tz[q])
					q--;
			}

		}

		delete[] sz;
		delete[] tz;

	}

	delete[] dt;

}

void DistanceTransform_signed(double* dt, int8_t* b, int32_t m, int32_t n) {

	int32_t inf = m+n;

	int32_t* g;
	g			=	new int32_t[m*n];

	int32_t* gc;
	gc			=	new int32_t[m*n];

#pragma omp parallel for shared(b,g,gc) schedule(dynamic)
	for (int32_t x=0; x<m; x++) {

		if (b[x] == 1) {
			g[x]  		= 0;
			gc[x] 		= inf;
		}
		else if (b[x] == -1) {
			g[x] 		= inf;
			gc[x]		= 0;
		}
		else {
			g[x]		= 0;
			gc[x]		= 0;
		}

		// Scan 1
		for (int32_t y=1; y<n; y++)	{
			if (b[x+y*m] == 1) {
				g[x+y*m]  		= 0;
				gc[x+y*m] 		= 1 + gc[x+(y-1)*m];
			}
			else if (b[x+y*m] == -1) {
				g[x+y*m] 		= 1 + g[x+(y-1)*m];
				gc[x+y*m]  	= 0;
			}
			else {
				g[x+y*m] 		= 0;
				gc[x+y*m] 		= 0;
			}
		}

		// Scan 2
		for (int32_t y=(n-2); y>= 0; y--)	{
			if (g[x+(y+1)*m] < g[x+y*m])
				g[x+y*m] = 1 + g[x+(y+1)*m];

			if (gc[x+(y+1)*m] < gc[x+y*m])
				gc[x+y*m] = 1 + gc[x+(y+1)*m];
		}

	}

	// Second phase
	int32_t* s;
	int32_t* t;
	int32_t q = 0;
	int32_t w;

	int32_t* sc;
	int32_t* tc;
	int32_t qc = 0;
	int32_t wc;

	int32_t dt_tmp;

#pragma omp parallel for shared(g,gc,dt) private(q,w,s,t,qc,wc,sc,tc) schedule(dynamic)
	for (int32_t y=0; y<n; y++)	{


		s			=	new int32_t[m];
		t			=	new int32_t[m];

		sc			=	new int32_t[m];
		tc			=	new int32_t[m];


		s[0] 		= 0;
		t[0] 		= 0;
		q 			= 0;

		sc[0] 	= 0;
		tc[0] 		= 0;
		qc 		= 0;

		// Scan 3
		for (int32_t x=1; x<m; x++)	{

			while (q >= 0 && ( F(t[q],s[q],g[s[q]+y*m]*g[s[q]+y*m]) > F(t[q],x,g[x+y*m]*g[x+y*m]) ) )
				q--;
			if (q < 0)	{
				q = 0;
				s[0] = x;
			}	else	{
				w = 1 + Sep(s[q],x,g[s[q]+y*m]*g[s[q]+y*m],g[x+y*m]*g[x+y*m]);
				if (w < m)	{
					q++;
					s[q] 	= x;
					t[q] 	= w;
				}
			}


			while (qc >= 0 && ( F(tc[qc],sc[qc],gc[sc[qc]+y*m]*gc[sc[qc]+y*m]) > F(tc[qc],x,gc[x+y*m]*gc[x+y*m]) ) )
				qc--;
			if (qc < 0)	{
				qc = 0;
				sc[0] = x;
			}	else	{
				wc = 1 + Sep(sc[qc],x,gc[sc[qc]+y*m]*gc[sc[qc]+y*m],gc[x+y*m]*gc[x+y*m]);
				if (wc < m)	{
					qc++;
					sc[qc] = x;
					tc[qc]  = wc;
				}
			}

		}

		// Scan 4
		for (int32_t x=(m-1); x>=0; x--)	{

			dt_tmp 	= F(x,sc[qc],gc[sc[qc]+y*m]*gc[sc[qc]+y*m]) - F(x,s[q],g[s[q]+y*m]*g[s[q]+y*m]);

			if (dt_tmp > 0)
				dt[x+y*m] = sqrt(dt_tmp);
			else if (dt_tmp == 0)
				dt[x+y*m] = 0;
			else
				dt[x+y*m] = -sqrt(-dt_tmp);

			if (x == t[q])
				q--;

			if (x == tc[qc])
				qc--;

		}

		delete[] s;
		delete[] t;

		delete[] sc;
		delete[] tc;

	}

	delete[] g;
	delete[] gc;


}

void DistanceTransform_signed(double* dtz, int8_t* b, int32_t m, int32_t n, int32_t p) {

	int32_t inf = m+n+p;

	int32_t* g;
	int32_t* gc;


	int32_t* dt;
	dt				=	new int32_t[m*n*p];

	int32_t* s;
	int32_t* t;

	int32_t q = 0;
	int32_t w;


	int32_t* dtc;
	dtc			=	new int32_t[m*n*p];

	int32_t* sc;
	int32_t* tc;

	int32_t qc 	= 0;
	int32_t wc;

#pragma omp parallel for shared(b,dt,dtc) private(g,gc,q,w,s,t,qc,wc,sc,tc) schedule(dynamic)
	for (int32_t z=0; z<p; z++) {

		g	=	new int32_t[m*n];
		gc	=	new int32_t[m*n];

		s	=	new int32_t[m];
		t	=	new int32_t[m];
		sc	=	new int32_t[m];
		tc	=	new int32_t[m];

		for (int32_t x=0; x<m; x++) {

			if (b[x+z*m*n] == -1) {
				g[x]  		= 0;
				gc[x] 		= inf;
			}
			else if (b[x+z*m*n] == 1) {
				g[x] 		= inf;
				gc[x]		= 0;
			}
			else {
				g[x]		= 0;
				gc[x]		= 0;
			}

			// Scan 1
			for (int32_t y=1; y<n; y++)	{
				if (b[x+y*m+z*m*n] == -1) {
					g[x+y*m]  		= 0;
					gc[x+y*m] 		= 1 + gc[x+(y-1)*m];
				}
				else if (b[x+y*m+z*m*n] == 1) {
					g[x+y*m] 		= 1 + g[x+(y-1)*m];
					gc[x+y*m]  	= 0;
				}
				else {
					g[x+y*m] 		= 0;
					gc[x+y*m] 		= 0;
				}
			}

			// Scan 2
			for (int32_t y=(n-2); y>= 0; y--)	{
				if (g[x+(y+1)*m] < g[x+y*m])
					g[x+y*m] = 1 + g[x+(y+1)*m];

				if (gc[x+(y+1)*m] < gc[x+y*m])
					gc[x+y*m] = 1 + gc[x+(y+1)*m];
			}

		}

		for (int32_t y=0; y<n; y++)	{

			q 		= 0;
			s[0] 	= 0;
			t[0] 	= 0;

			qc 	= 0;
			sc[0]	= 0;
			tc[0] 	= 0;

			// Scan 3
			for (int32_t x=1; x<m; x++)	{

				while (q >= 0 && ( F(t[q],s[q],g[s[q]+y*m]*g[s[q]+y*m]) > F(t[q],x,g[x+y*m]*g[x+y*m])) )
					q--;
				if (q < 0)	{
					q = 0;
					s[0] = x;
				}	else	{
					w = 1 + Sep(s[q],x,g[s[q]+y*m]*g[s[q]+y*m],g[x+y*m]*g[x+y*m]);
					if (w < m)	{
						q++;
						s[q] = x;
						t[q] = w;
					}
				}

				while (qc >= 0 && ( F(tc[qc],sc[qc],gc[sc[qc]+y*m]*gc[sc[qc]+y*m]) > F(tc[qc],x,gc[x+y*m]*gc[x+y*m]) ) )
					qc--;
				if (qc < 0)	{
					qc = 0;
					sc[0] = x;
				}	else	{
					wc = 1 + Sep(sc[qc],x,gc[sc[qc]+y*m]*gc[sc[qc]+y*m],gc[x+y*m]*gc[x+y*m]);
					if (wc < m)	{
						qc++;
						sc[qc] = x;
						tc[qc] = wc;
					}
				}

			}

			// Scan 4
			for (int32_t x=(m-1); x>=0; x--)	{

				dt[x+y*m+z*m*n] = F(x,s[q],g[s[q]+y*m]*g[s[q]+y*m]);
				if (x == t[q])
					q--;

				dtc[x+y*m+z*m*n] = F(x,sc[qc],gc[sc[qc]+y*m]*gc[sc[qc]+y*m]);
				if (x == tc[qc])
					qc--;

			}

		}

		delete[] g;
		delete[] gc;

		delete[] s;
		delete[] t;
		delete[] sc;
		delete[] tc;

	}

	int32_t* sz;
	int32_t* tz;
	int32_t qz 	= 0;
	int32_t wz;


	int32_t* szc;
	int32_t* tzc;
	int32_t qzc = 0;
	int32_t wzc;

#pragma omp parallel for shared(dt,dtc,dtz) private(qz,wz,sz,tz,qzc,wzc,szc,tzc) schedule(dynamic)
	for (int32_t x=0; x<m; x++)	{

		sz		=	new int32_t[p];
		tz		=	new int32_t[p];
		szc	=	new int32_t[p];
		tzc		=	new int32_t[p];

		for (int32_t y=0; y<n; y++)	{

			qz		= 0;
			sz[0]	= 0;
			tz[0] 	= 0;

			qzc 	= 0;
			szc[0]	= 0;
			tzc[0] 	= 0;

			// Scan 5
			for (int32_t z=1; z<p; z++)	{

				while (qz >= 0 && ( F(tz[qz],sz[qz],dt[x+y*m+sz[qz]*m*n]) > F(tz[qz],z,dt[x+y*m+z*m*n]) ) )
					qz--;
				if (qz < 0)	{
					qz = 0;
					sz[0] = z;
				}	else	{
					wz = 1 + Sep(sz[qz],z,dt[x+y*m+sz[qz]*m*n],dt[x+y*m+z*m*n]);
					if (wz < p)	{
						qz++;
						sz[qz] = z;
						tz[qz] = wz;
					}
				}

				while (qzc >= 0 && ( F(tzc[qzc],szc[qzc],dtc[x+y*m+szc[qzc]*m*n]) > F(tzc[qzc],z,dtc[x+y*m+z*m*n]) ) )
					qzc--;
				if (qzc < 0)	{
					qzc = 0;
					szc[0] = z;
				}	else	{
					wzc = 1 + Sep(szc[qzc],z,dtc[x+y*m+szc[qzc]*m*n],dtc[x+y*m+z*m*n]);
					if (wzc < p)	{
						qzc++;
						szc[qzc] = z;
						tzc[qzc] = wzc;
					}
				}



			}

			// Scan 6
			for (int32_t z=(p-1); z>=0; z--)	{

				dtz[x+y*m+z*m*n] = F(z,sz[qz],dt[x+y*m+sz[qz]*m*n]) - F(z,szc[qzc],dtc[x+y*m+szc[qzc]*m*n]);

				if (dtz[x+y*m+z*m*n] > 0)
					dtz[x+y*m+z*m*n] = sqrt(dtz[x+y*m+z*m*n]);
				else if (dtz[x+y*m+z*m*n] == 0)
					dtz[x+y*m+z*m*n] = 0;
				else
					dtz[x+y*m+z*m*n] = -sqrt(-dtz[x+y*m+z*m*n]);

				if (z == tz[qz])
					qz--;
				if (z == tzc[qzc])
					qzc--;
			}

		}

		delete[] sz;
		delete[] tz;
		delete[] szc;
		delete[] tzc;


	}

	delete[] dt;
	delete[] dtc;

}
