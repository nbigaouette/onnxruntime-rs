//! Module containing tensor with memory owned by the ONNX Runtime

use std::{fmt::Debug, ops::Deref, ptr, rc, result};

use ndarray::ArrayView;
use thiserror::Error;
use tracing::debug;

use onnxruntime_sys as sys;

use crate::{
    error::call_ort,
    g_ort,
    memory::MemoryInfo,
    tensor::{TensorData, TensorDataToType, TensorElementDataType},
    OrtError,
};

/// Errors that can occur while extracting a tensor from ort output.
#[derive(Error, Debug)]
pub enum TensorExtractError {
    /// The user tried to extract the wrong type of tensor from the underlying data
    #[error(
        "Data type mismatch: was {:?}, tried to convert to {:?}",
        actual,
        requested
    )]
    DataTypeMismatch {
        /// The actual type of the ort output
        actual: TensorElementDataType,
        /// The type corresponding to the attempted conversion into a Rust type, not equal to `actual`
        requested: TensorElementDataType,
    },
    /// An onnxruntime error occurred
    #[error("Onnxruntime error: {:?}", 0)]
    OrtError(#[from] OrtError),
}

/// A wrapper around a tensor produced by onnxruntime inference.
///
/// Since different outputs for the same model can have different types, this type is used to allow
/// the user to dynamically query each output's type and extract the appropriate tensor type with
/// [try_extract].
#[derive(Debug)]
pub struct DynOrtTensor<'m, D>
where
    D: ndarray::Dimension,
{
    // TODO could this also hold a Vec<u8> for strings so that the extracted tensor could then
    // hold a Vec<&str>?
    tensor_ptr_holder: rc::Rc<TensorPointerHolder>,
    memory_info: &'m MemoryInfo,
    shape: D,
    tensor_element_len: usize,
    data_type: TensorElementDataType,
}

impl<'m, D> DynOrtTensor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) fn new(
        tensor_ptr: *mut sys::OrtValue,
        memory_info: &'m MemoryInfo,
        shape: D,
        tensor_element_len: usize,
        data_type: TensorElementDataType,
    ) -> DynOrtTensor<'m, D> {
        DynOrtTensor {
            tensor_ptr_holder: rc::Rc::from(TensorPointerHolder { tensor_ptr }),
            memory_info,
            shape,
            tensor_element_len,
            data_type,
        }
    }

    /// The ONNX data type this tensor contains.
    pub fn data_type(&self) -> TensorElementDataType {
        self.data_type
    }

    /// Extract a tensor containing `T`.
    ///
    /// Where the type permits it, the tensor will be a view into existing memory.
    ///
    /// # Errors
    ///
    /// An error will be returned if `T`'s ONNX type doesn't match this tensor's type, or if an
    /// onnxruntime error occurs.
    pub fn try_extract<'t, T>(&self) -> result::Result<OrtOwnedTensor<'t, T, D>, TensorExtractError>
    where
        T: TensorDataToType + Clone + Debug,
        'm: 't, // mem info outlives tensor
        D: 't,  // not clear why this is needed since we clone the shape, but it doesn't make
                // a difference in practice since the shape is extracted from the tensor
    {
        if self.data_type != T::tensor_element_data_type() {
            Err(TensorExtractError::DataTypeMismatch {
                actual: self.data_type,
                requested: T::tensor_element_data_type(),
            })
        } else {
            // Note: Both tensor and array will point to the same data, nothing is copied.
            // As such, there is no need to free the pointer used to create the ArrayView.
            assert_ne!(self.tensor_ptr_holder.tensor_ptr, ptr::null_mut());

            let mut is_tensor = 0;
            unsafe {
                call_ort(|ort| {
                    ort.IsTensor.unwrap()(self.tensor_ptr_holder.tensor_ptr, &mut is_tensor)
                })
            }
            .map_err(OrtError::IsTensor)?;
            assert_eq!(is_tensor, 1);

            let data = T::extract_data(
                self.shape.clone(),
                self.tensor_element_len,
                rc::Rc::clone(&self.tensor_ptr_holder),
            )?;

            Ok(OrtOwnedTensor { data })
        }
    }
}

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
///
/// The tensor hosts an [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
/// of the data on the C side. This allows manipulation on the Rust side using `ndarray` without copying the data.
///
/// `OrtOwnedTensor` implements the [`std::deref::Deref`](#impl-Deref) trait for ergonomic access to
/// the underlying [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
#[derive(Debug)]
pub struct OrtOwnedTensor<'t, T, D>
where
    T: TensorDataToType,
    D: ndarray::Dimension,
{
    data: TensorData<'t, T, D>,
}

impl<'t, T, D> OrtOwnedTensor<'t, T, D>
where
    T: TensorDataToType,
    D: ndarray::Dimension + 't,
{
    /// Produce a [ViewHolder] for the underlying data, which
    pub fn view<'s>(&'s self) -> ViewHolder<'s, T, D>
    where
        't: 's, // tensor ptr can outlive the TensorData
    {
        ViewHolder::new(&self.data)
    }
}

/// An intermediate step on the way to an ArrayView.
// Since Deref has to produce a reference, and the referent can't be a local in deref(), it must
// be a field in a struct. This struct exists only to hold that field.
// Its lifetime 's is bound to the TensorData its view was created around, not the underlying tensor
// pointer, since in the case of strings the data is the Array in the TensorData, not the pointer.
pub struct ViewHolder<'s, T, D>
where
    T: TensorDataToType,
    D: ndarray::Dimension,
{
    array_view: ndarray::ArrayView<'s, T, D>,
}

impl<'s, T, D> ViewHolder<'s, T, D>
where
    T: TensorDataToType,
    D: ndarray::Dimension,
{
    fn new<'t>(data: &'s TensorData<'t, T, D>) -> ViewHolder<'s, T, D>
    where
        't: 's, // underlying tensor ptr lives at least as long as TensorData
    {
        match data {
            TensorData::TensorPtr { array_view, .. } => ViewHolder {
                // we already have a view, but creating a view from a view is cheap
                array_view: array_view.view(),
            },
            TensorData::Strings { strings } => ViewHolder {
                // This view creation has to happen here, not at new()'s callsite, because
                // a field can't be a reference to another field in the same struct. Thus, we have
                // this separate struct to hold the view that refers to the `Array`.
                array_view: strings.view(),
            },
        }
    }
}

impl<'t, T, D> Deref for ViewHolder<'t, T, D>
where
    T: TensorDataToType,
    D: ndarray::Dimension,
{
    type Target = ArrayView<'t, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array_view
    }
}

/// Holds on to a tensor pointer until dropped.
///
/// This allows creating an [OrtOwnedTensor] from a [DynOrtTensor] without consuming `self`, which
/// would prevent retrying extraction and also make interacting with outputs `Vec` awkward.
/// It also avoids needing `OrtOwnedTensor` to keep a reference to `DynOrtTensor`, which would be
/// inconvenient.
#[derive(Debug)]
pub struct TensorPointerHolder {
    pub(crate) tensor_ptr: *mut sys::OrtValue,
}

impl Drop for TensorPointerHolder {
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping OrtOwnedTensor.");
        unsafe { g_ort().ReleaseValue.unwrap()(self.tensor_ptr) }

        self.tensor_ptr = ptr::null_mut();
    }
}
