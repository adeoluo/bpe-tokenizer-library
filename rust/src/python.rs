//! PyO3 bindings, compiled only with the `python` feature (enabled by maturin).

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::tokenizer::Tokenizer;
use crate::unicode_tables::{default_table, table_for, ClassTable, TABLES};

/// Resolve an optional Unicode version to its character-class table.
fn classes(unicode_version: Option<&str>) -> PyResult<ClassTable> {
    match unicode_version {
        None => Ok(default_table()),
        Some(version) => table_for(version).ok_or_else(|| {
            PyValueError::new_err(format!("no character table for Unicode {version}"))
        }),
    }
}

/// A byte-level BPE tokenizer backed by the Rust core.
#[pyclass(name = "Tokenizer", module = "bpe_tokenizer_rs", frozen)]
struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(signature = (merges, unicode_version=None))]
    fn new(merges: Vec<(String, String)>, unicode_version: Option<&str>) -> PyResult<Self> {
        Ok(PyTokenizer {
            inner: Tokenizer::from_merges_with(merges, classes(unicode_version)?),
        })
    }

    /// Encode `text` into token ids. Releases the GIL while encoding.
    fn encode(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.inner.encode(text))
    }

    /// Encode several texts. Releases the GIL while encoding.
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<u32>> {
        py.detach(|| texts.iter().map(|t| self.inner.encode(t)).collect())
    }

    /// Decode token ids; raises `KeyError` for an unknown id.
    fn decode(&self, ids: Vec<i64>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|err| PyKeyError::new_err(err.0))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    fn merges(&self) -> Vec<(String, String)> {
        self.inner.merges().to_vec()
    }
}

/// Split `text` into GPT-2 style chunks.
#[pyfunction]
#[pyo3(signature = (text, unicode_version=None))]
fn pretokenize<'a>(text: &'a str, unicode_version: Option<&str>) -> PyResult<Vec<&'a str>> {
    Ok(crate::pretokenize::pretokenize_with(
        text,
        classes(unicode_version)?,
    ))
}

/// Train merges on `corpus` for a target vocabulary size.
#[pyfunction]
#[pyo3(signature = (corpus, vocab_size, unicode_version=None))]
fn train_bpe(
    py: Python<'_>,
    corpus: Vec<String>,
    vocab_size: i64,
    unicode_version: Option<&str>,
) -> PyResult<Vec<(String, String)>> {
    let table = classes(unicode_version)?;
    Ok(py.detach(|| crate::trainer::train_bpe_with(&corpus, vocab_size, table)))
}

#[pymodule]
fn bpe_tokenizer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(pretokenize, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add("UNICODE_VERSION", crate::UNICODE_VERSION)?;
    let versions: Vec<&str> = TABLES.iter().map(|&(v, _)| v).collect();
    m.add("SUPPORTED_UNICODE_VERSIONS", versions)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
