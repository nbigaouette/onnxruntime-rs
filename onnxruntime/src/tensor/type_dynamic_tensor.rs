//! Module containing tensor enums allowing for multiple input types simultaneously.

use crate::memory::MemoryInfo;
use crate::ndarray::Array;
use crate::tensor::OrtTensor;
use crate::Result;
use std::fmt::Debug;

use onnxruntime_sys as sys;

/// Trait used for constructing inputs with multiple element types from [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
pub trait FromArray<T, D: ndarray::Dimension> {
    /// Wrap [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) into enum with specific dtype variants.
    fn from_array(array: Array<T, D>) -> InputTensor<D>;
}

macro_rules! impl_convert_trait {
    ($type_:ty, $variant:expr) => {
        impl<D: ndarray::Dimension> FromArray<$type_, D> for InputTensor<D> {
            fn from_array(array: Array<$type_, D>) -> InputTensor<D> {
                $variant(array)
            }
        }
    };
}

/// Input tensor enum with tensor element type as a variant.
///
/// Required for supplying inputs with different types
#[derive(Debug)]
#[allow(missing_docs)]
pub enum InputTensor<D: ndarray::Dimension> {
    FloatTensor(Array<f32, D>),
    Uint8Tensor(Array<u8, D>),
    Int8Tensor(Array<i8, D>),
    Uint16Tensor(Array<u16, D>),
    Int16Tensor(Array<i16, D>),
    Int32Tensor(Array<i32, D>),
    Int64Tensor(Array<i64, D>),
    DoubleTensor(Array<f64, D>),
    Uint32Tensor(Array<u32, D>),
    Uint64Tensor(Array<u64, D>),
    StringTensor(Array<String, D>),
}

/// This tensor is used to copy an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// from InputTensor to the runtime's memory with support to multiple input tensor types.
///
/// **NOTE**: The type is not meant to be used directly, use an InputTensor constructed from
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) instead.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum InputOrtTensor<'t, D: ndarray::Dimension> {
    FloatTensor(OrtTensor<'t, f32, D>),
    Uint8Tensor(OrtTensor<'t, u8, D>),
    Int8Tensor(OrtTensor<'t, i8, D>),
    Uint16Tensor(OrtTensor<'t, u16, D>),
    Int16Tensor(OrtTensor<'t, i16, D>),
    Int32Tensor(OrtTensor<'t, i32, D>),
    Int64Tensor(OrtTensor<'t, i64, D>),
    DoubleTensor(OrtTensor<'t, f64, D>),
    Uint32Tensor(OrtTensor<'t, u32, D>),
    Uint64Tensor(OrtTensor<'t, u64, D>),
    StringTensor(OrtTensor<'t, String, D>),
}

impl<D: ndarray::Dimension> InputTensor<D> {
    /// Get shape of the underlying array.
    pub fn shape(&self) -> &[usize] {
        match self {
            InputTensor::FloatTensor(x) => x.shape(),
            InputTensor::Uint8Tensor(x) => x.shape(),
            InputTensor::Int8Tensor(x) => x.shape(),
            InputTensor::Uint16Tensor(x) => x.shape(),
            InputTensor::Int16Tensor(x) => x.shape(),
            InputTensor::Int32Tensor(x) => x.shape(),
            InputTensor::Int64Tensor(x) => x.shape(),
            InputTensor::DoubleTensor(x) => x.shape(),
            InputTensor::Uint32Tensor(x) => x.shape(),
            InputTensor::Uint64Tensor(x) => x.shape(),
            InputTensor::StringTensor(x) => x.shape(),
        }
    }
}

impl_convert_trait!(f32, InputTensor::FloatTensor);
impl_convert_trait!(u8, InputTensor::Uint8Tensor);
impl_convert_trait!(i8, InputTensor::Int8Tensor);
impl_convert_trait!(u16, InputTensor::Uint16Tensor);
impl_convert_trait!(i16, InputTensor::Int16Tensor);
impl_convert_trait!(i32, InputTensor::Int32Tensor);
impl_convert_trait!(i64, InputTensor::Int64Tensor);
impl_convert_trait!(f64, InputTensor::DoubleTensor);
impl_convert_trait!(u32, InputTensor::Uint32Tensor);
impl_convert_trait!(u64, InputTensor::Uint64Tensor);
impl_convert_trait!(String, InputTensor::StringTensor);

impl<'t, D: ndarray::Dimension> InputOrtTensor<'t, D> {
    pub(crate) fn from_input_tensor<'m>(
        memory_info: &'m MemoryInfo,
        allocator_ptr: *mut sys::OrtAllocator,
        input_tensor: InputTensor<D>,
    ) -> Result<InputOrtTensor<'t, D>>
    where
        'm: 't,
    {
        match input_tensor {
            InputTensor::FloatTensor(array) => Ok(InputOrtTensor::FloatTensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Uint8Tensor(array) => Ok(InputOrtTensor::Uint8Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Int8Tensor(array) => Ok(InputOrtTensor::Int8Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Uint16Tensor(array) => Ok(InputOrtTensor::Uint16Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Int16Tensor(array) => Ok(InputOrtTensor::Int16Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Int32Tensor(array) => Ok(InputOrtTensor::Int32Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Int64Tensor(array) => Ok(InputOrtTensor::Int64Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::DoubleTensor(array) => Ok(InputOrtTensor::DoubleTensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Uint32Tensor(array) => Ok(InputOrtTensor::Uint32Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::Uint64Tensor(array) => Ok(InputOrtTensor::Uint64Tensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
            InputTensor::StringTensor(array) => Ok(InputOrtTensor::StringTensor(
                OrtTensor::from_array(memory_info, allocator_ptr, array)?,
            )),
        }
    }

    pub(crate) fn c_ptr(&self) -> *const sys::OrtValue {
        match self {
            InputOrtTensor::FloatTensor(x) => x.c_ptr,
            InputOrtTensor::Uint8Tensor(x) => x.c_ptr,
            InputOrtTensor::Int8Tensor(x) => x.c_ptr,
            InputOrtTensor::Uint16Tensor(x) => x.c_ptr,
            InputOrtTensor::Int16Tensor(x) => x.c_ptr,
            InputOrtTensor::Int32Tensor(x) => x.c_ptr,
            InputOrtTensor::Int64Tensor(x) => x.c_ptr,
            InputOrtTensor::DoubleTensor(x) => x.c_ptr,
            InputOrtTensor::Uint32Tensor(x) => x.c_ptr,
            InputOrtTensor::Uint64Tensor(x) => x.c_ptr,
            InputOrtTensor::StringTensor(x) => x.c_ptr,
        }
    }
}
