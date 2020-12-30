#//#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>

/* Docstrings */
static char module_docstring[] =
    "This module is designed to return a rendered SPH output based on input particle coordinates and smoothing lengths The sph_A array should contain extra sph output information if output is a quantity apart from density";
static char pyjack_docstring[] =
    "Arguments: sph_x, sph_y, sph_h, sph_A, nbins, lower, upper, kernel_n, M_sph";

/* Available functions */
static PyObject *smoother_pyjack(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"smoother", smoother_pyjack, METH_VARARGS, pyjack_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC initpyjack(void)
{
    PyObject *m = Py_InitModule3("pyjack", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *smoother_pyjack(PyObject *self, PyObject *args)
{
    int i, j, k, nbins;
    double kernel_n, upper, lower, M_sph;
    PyObject *sph_x_obj, *sph_y_obj, *sph_h_obj, *sph_A_obj;
    printf("Function Start");
    
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOidddd", &sph_x_obj, &sph_y_obj, &sph_h_obj, &sph_A_obj,
			  &nbins, &lower, &upper, &kernel_n, &M_sph))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *sph_x_array = PyArray_FROM_OTF(sph_x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *sph_y_array = PyArray_FROM_OTF(sph_y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *sph_h_array = PyArray_FROM_OTF(sph_h_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *sph_A_array = PyArray_FROM_OTF(sph_A_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    /* If that didn't work, throw an exception. */
    if (sph_x_array == NULL || sph_y_array == NULL || sph_h_array == NULL || sph_A_array == NULL) {
        Py_XDECREF(sph_x_array);
        Py_XDECREF(sph_y_array);
	Py_XDECREF(sph_h_array);
	Py_XDECREF(sph_A_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(sph_x_array, 0);
    int ntot = nbins*nbins;
    
    /* Get pointers to the data as C-types. */
    double *sph_x    = (double*)PyArray_DATA(sph_x_array);
    double *sph_y    = (double*)PyArray_DATA(sph_y_array);
    double *sph_h    = (double*)PyArray_DATA(sph_h_array);
    double *sph_A    = (double*)PyArray_DATA(sph_A_array);
    double *rendered = (double*)malloc(2*ntot*sizeof(double));

    /* Do the computation */
    double width = (upper-lower)/nbins;
    double * bins = malloc(nbins*sizeof(double));
    for (i=0; i<nbins; i++) bins[i] = lower+(i+0.5)*width;
    for (i=0; i<2*nbins; i++) rendered[i] = 0;
    
    //printf("Just before For loop");
    for (i=0; i<N; i++) {
      /* Search in sph h Radius Bounding Boxes. What about edges? */
        double h2 = sph_h[i]*sph_h[i];
	double one_h2 = 1 / h2;
        double x_pos = sph_x[i];
        double y_pos = sph_y[i];
	double A_val = sph_A[i];
	
        int mx_lower = (int)floor((x_pos - lower - sph_h[i])/width - 0.5);
        int mx_upper = (int)floor((x_pos - lower + sph_h[i])/width + 0.5);
        if (mx_lower<0) mx_lower=0;
        if (mx_upper>nbins-1) mx_upper=nbins-1;

        int my_lower = (int)floor((y_pos - lower - sph_h[i])/width - 0.5);
        int my_upper = (int)floor((y_pos - lower + sph_h[i])/width + 0.5);
        if (my_lower<0) my_lower=0;
        if (my_upper>nbins-1) my_upper=nbins-1;

        for (j=mx_lower; j<mx_upper; j++) {
            double dist_x = (x_pos-bins[j])*(x_pos-bins[j]);
            for (k=my_lower; k<my_upper; k++) {
	        double dist2 = dist_x + (y_pos-bins[k])*(y_pos-bins[k]);
                if (dist2<h2) {
		  double kern =  (1 - dist2 * one_h2)*(1- dist2*one_h2);
		  double temp = kern*kern * one_h2;
		  rendered[j+nbins*k]      += temp;
		  rendered[ntot+j+nbins*k] += temp * A_val;
		}
		

            }
        }

    }

    for (j=0; j<ntot; j++) {
        rendered[j]      *= (kernel_n+1) / M_PI * M_sph;
	rendered[ntot+j] *= (kernel_n+1) / M_PI * M_sph;
    }

    free(bins);
    
    /* Clean up. */
    Py_XDECREF(sph_x_array);
    Py_XDECREF(sph_y_array);
    Py_XDECREF(sph_h_array);
    Py_XDECREF(sph_A_array);
      
    /* Build the output tuple */
    npy_intp dims[1] = {2*ntot};
    PyObject *rendered_return  = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA(rendered_return), rendered, sizeof(double)*2*nbins*nbins);

    free(rendered);

    return rendered_return;
}
