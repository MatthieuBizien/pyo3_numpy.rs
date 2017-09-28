pub extern crate pyo3;

use std::mem::size_of;
use std::slice::from_raw_parts_mut;

use pyo3::prelude::*;

// A link between a rust object and a numpy dtype
pub trait NumpyDtype {
    // The dtype name, like "float32"
    const DTYPE: &'static str;
}

// Numpy don't have u8 or i8 🙁

impl NumpyDtype for u16 {
    const DTYPE: &'static str = "uint16";
}

impl NumpyDtype for i16 {
    const DTYPE: &'static str = "int16";
}

// Rust don't have f16 😨

impl NumpyDtype for u32 {
    const DTYPE: &'static str = "uint32";
}

impl NumpyDtype for i32 {
    const DTYPE: &'static str = "int32";
}

impl NumpyDtype for f32 {
    const DTYPE: &'static str = "float32";
}

impl NumpyDtype for u64 {
    const DTYPE: &'static str = "uint64";
}

impl NumpyDtype for i64 {
    const DTYPE: &'static str = "int64";
}

impl NumpyDtype for f64 {
    const DTYPE: &'static str = "float64";
}

// Numpy don't have i128 or u128 🙁
// Rust don't have f128 😨

// Convert a Rust object to a numpy object
pub trait ToNumpy {
    // Convert a Rust object to a numpy object (consumme self)
    fn to_numpy(self, py: Python) -> PyResult<PyObject>;
}

impl<'a, T: NumpyDtype> ToNumpy for &'a mut Vec<T> {
    fn to_numpy(self, py: Python) -> PyResult<PyObject> {
        let locals = PyDict::new(py);

        // We populate the python namespace
        let frombuffer = py.import("numpy")?.get("frombuffer")?;
        locals.set_item("frombuffer", frombuffer)?;
        locals.set_item("dtype", PyString::new(py, T::DTYPE))?;

        // We convert the array to python bytes
        let nbytes = self.len() * size_of::<T>();
        let val = unsafe {
            // We have to use unsafe to convert an array of T to bytes
            from_raw_parts_mut(self.as_mut_ptr() as *mut u8, nbytes)
        };
        locals.set_item("x", PyBytes::new(py, val))?;

        // We can now create the numpy array from the bytes
        let o = py.eval("frombuffer(x, dtype=dtype)", None, Some(locals))?;
        let o = PyObject::from_borrowed_ptr_or_err(py, o.as_ptr())?;
        Ok(o)
    }
}

#[cfg(test)]
mod tests {
    use std;
    use pyo3::prelude::*;
    use super::*;

    trait OrExit<T> {
        fn unwrap_or_exit(self, msg: &str, py: Python) -> T;
    }

    impl<T> OrExit<T> for PyResult<T> {
        fn unwrap_or_exit(self, msg: &str, py: Python) -> T {
            self.unwrap_or_else(|err| {
                err.print(py);
                let mut msg = String::from(msg);
                let locals = PyDict::new(py);
                if let Ok(sys) = py.import("sys") {
                    if let Ok(()) = sys.set_item("sys", sys) {
                        if let Ok(()) = py.run("print('Python:', sys.version)", None, Some(locals))
                        {
                        } else {
                            msg.push_str(" + NO PRINT");
                        }
                    } else {
                        msg.push_str(" + NO SET SYS");
                    }
                } else {
                    msg.push_str(" + NO SYS");
                }
                panic!(msg);
            })
        }
    }

    fn test_vec_to_numpy_<T>(v: &mut Vec<T>)
    where
        T: std::fmt::Debug,
        T: NumpyDtype,
    {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let locals = PyDict::new(py);

        let _ = py.import("numpy")
            .unwrap_or_exit("Impossible to load numpy", py);
        let py_v = v.to_numpy(py).unwrap_or_exit("v.to_numpy(py)", py);
        locals.set_item("v", py_v).unwrap_or_exit("v=", py);
        let sum_v = py.eval("float(sum(v * 2))", None, Some(locals))
            .unwrap_or_exit("calc", py);
        let sum_v: f32 = sum_v.extract().unwrap_or_exit("extract_sum", py);
        assert_eq!(sum_v, 12.);
        let dtype = py.eval("str(v.dtype)", None, Some(locals))
            .unwrap_or_exit("dtype", py);
        let dtype: String = dtype.extract().unwrap_or_exit("extract dtype", py);
        assert_eq!(dtype, T::DTYPE);
    }

    #[test]
    fn test_vec_to_numpy() {
        let mut v: Vec<i16> = vec![1, 2, 3];
        test_vec_to_numpy_(&mut v);
        let mut v: Vec<u16> = vec![1, 2, 3];
        test_vec_to_numpy_(&mut v);
        let mut v: Vec<i32> = vec![1, 2, 3];
        test_vec_to_numpy_(&mut v);
        let mut v: Vec<u32> = vec![1, 2, 3];
        test_vec_to_numpy_(&mut v);
        let mut v: Vec<i64> = vec![1, 2, 3];
        test_vec_to_numpy_(&mut v);
        let mut v: Vec<u64> = vec![1, 2, 3];
        test_vec_to_numpy_(&mut v);

        save_csv(vec![1, 2, 3], "/tmp/myfile.csv").unwrap()
    }

    fn save_csv(mut v: Vec<u32>, path: &str) -> PyResult<()> {
        // We create and populate the namespace
        let gil = Python::acquire_gil();
        let py = gil.python();
        let locals = PyDict::new(py);
        let numpy = py.import("numpy")?;
        locals.set_item("np", numpy)?;

        // Use to_numpy to send an array to numpy
        let myarray = v.to_numpy(py)?;
        locals.set_item("myarray", myarray)?;

        // The dtype is adapted
        let dtype = py.eval("str(myarray.dtype)", None, Some(locals))?;
        let dtype: String = dtype.extract().unwrap_or("NOT A STRING".to_string());
        assert_eq!(dtype, "uint32");

        // We can use python and numpy functions
        locals.set_item("path", path)?;
        py.run("np.savetxt(path, myarray)", None, Some(locals))?;
        Ok(())
    }
}
