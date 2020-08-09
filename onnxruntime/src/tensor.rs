use std::{fmt::Debug, ops::Deref};

use ndarray::{Array, ArrayView};

use onnxruntime_sys as sys;

use crate::{
    error::status_to_result, g_ort, memory::MemoryInfo, OrtError, Result,
    TypeToTensorElementDataType,
};

#[derive(Debug)]
pub struct Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
{
    pub(crate) c_ptr: *mut sys::OrtValue,
    array: Array<T, D>,
    memory_info: &'t MemoryInfo,
}

impl<'t, T, D> Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
{
    pub(crate) fn from_array<'m>(
        memory_info: &'m MemoryInfo,
        mut array: Array<T, D>,
    ) -> Result<Tensor<'t, T, D>>
    where
        'm: 't, // 'm outlives 't
    {
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;
        let tensor_values_ptr: *mut std::ffi::c_void = array.as_mut_ptr() as *mut std::ffi::c_void;
        assert_ne!(tensor_values_ptr, std::ptr::null_mut());

        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = array.shape().len() as u64;

        let status = unsafe {
            (*g_ort()).CreateTensorWithDataAsOrtValue.unwrap()(
                memory_info.ptr,
                tensor_values_ptr,
                (array.len() * std::mem::size_of::<T>()) as u64,
                shape_ptr,
                shape_len,
                T::tensor_element_data_type() as u32,
                tensor_ptr_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::CreateTensorWithData)?;
        assert_ne!(tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { (*g_ort()).IsTensor.unwrap()(tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_eq!(is_tensor, 1);

        Ok(Tensor {
            c_ptr: tensor_ptr,
            array,
            memory_info,
        })
    }
}

impl<'t, T, D> Deref for Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
{
    type Target = Array<T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<'t, T, D> Drop for Tensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
{
    fn drop(&mut self) {
        // We need to let the C part free
        println!("Dropping Tensor.");
        unsafe { (*g_ort()).ReleaseValue.unwrap()(self.c_ptr) }

        self.c_ptr = std::ptr::null_mut();
    }
}

#[derive(Debug)]
pub struct TensorFromOrt<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    array_view: ArrayView<'t, T, D>,
    memory_info: &'m MemoryInfo,
}

impl<'t, 'm, T, D> Deref for TensorFromOrt<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
{
    type Target = ArrayView<'t, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array_view
    }
}

#[derive(Debug)]
pub struct TensorFromOrtExtractor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) tensor_ptr: *mut sys::OrtValue,
    memory_info: &'m MemoryInfo,
    shape: D,
}

impl<'m, D> TensorFromOrtExtractor<'m, D>
where
    D: ndarray::Dimension,
{
    pub(crate) fn new(memory_info: &'m MemoryInfo, shape: D) -> TensorFromOrtExtractor<'m, D> {
        TensorFromOrtExtractor {
            tensor_ptr: std::ptr::null_mut(),
            memory_info,
            shape,
        }
    }

    pub(crate) fn extract<'t, T>(self) -> Result<TensorFromOrt<'t, 'm, T, D>>
    where
        T: TypeToTensorElementDataType + Debug,
    {
        // Note: Both tensor and array will point to the same data, nothing is copied.
        // As such, there is no need too free the pointer used to create the ArrayView.

        assert_ne!(self.tensor_ptr, std::ptr::null_mut());

        let mut is_tensor = 0;
        let status = unsafe { (*g_ort()).IsTensor.unwrap()(self.tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_eq!(is_tensor, 1);

        // Get pointer to output tensor float values
        let mut output_array_ptr: *mut T = std::ptr::null_mut();
        let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
        let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void =
            output_array_ptr_ptr as *mut *mut std::ffi::c_void;
        let status = unsafe {
            (*g_ort()).GetTensorMutableData.unwrap()(self.tensor_ptr, output_array_ptr_ptr_void)
        };
        status_to_result(status).map_err(OrtError::IsTensor)?;
        assert_ne!(output_array_ptr, std::ptr::null_mut());

        let array_view = unsafe { ArrayView::from_shape_ptr(self.shape, output_array_ptr) };

        Ok(TensorFromOrt {
            tensor_ptr: self.tensor_ptr,
            array_view,
            memory_info: self.memory_info,
        })
    }
}

impl<'t, 'm, T, D> Drop for TensorFromOrt<'t, 'm, T, D>
where
    T: TypeToTensorElementDataType + Debug,
    D: ndarray::Dimension,
    'm: 't, // 'm outlives 't
{
    fn drop(&mut self) {
        println!("Dropping TensorFromOrt.");
        unsafe { (*g_ort()).ReleaseValue.unwrap()(self.tensor_ptr) }

        self.tensor_ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AllocatorType, MemType};
    use ndarray::{arr0, arr1, arr2, arr3};

    #[test]
    fn tensor_from_array_0d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr0::<i32>(123);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[]);
    }

    #[test]
    fn tensor_from_array_1d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[6]);
    }

    #[test]
    fn tensor_from_array_2d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[2, 6]);
    }

    #[test]
    fn tensor_from_array_3d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            [[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
        ]);
        let tensor = Tensor::from_array(&memory_info, array).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 6]);
    }
}
