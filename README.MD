# Py03_Numpy

Efficient bindings between Rust and Numpy with PyO3

# Usage

To use Py03_Numpy, add to `Cargo.toml`

```ini
[dependencies]
pyo3_numpy = "0.2"
```

Example for saving a csv
```rust
extern create pyo3_numpy;
use pyo3_numpy::pyo3::Python;

fn main() {
    save_csv(vec![1, 2, 3], "/tmp/myfile.csv").unwrap();
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
```