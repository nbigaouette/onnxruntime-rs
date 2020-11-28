use std::ffi::CString;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use ndarray::{Array, Dimension, IxDyn};

use onnxruntime_sys as sys;

use crate::error::{status_to_result, OrtError};
use crate::memory::MemoryInfo;
use crate::session::{Output, Session};
use crate::tensor::OrtTensor;
use crate::{g_ort, Result, TypeToTensorElementDataType};

pub trait Element: 'static + Clone + Debug + TypeToTensorElementDataType + Default {}

impl<T: 'static + Clone + Debug + TypeToTensorElementDataType + Default> Element for T {}

fn names_to_ptrs(names: impl Iterator<Item = String>) -> Vec<*const i8> {
    names
        .map(|name| CString::new(name.clone()).unwrap().into_raw() as *const _)
        .collect()
}

fn compute_output_shapes<TIn, DIn: Dimension>(
    input_arrays: &[Array<TIn, DIn>],
    outputs: &[Output],
) -> Vec<Vec<usize>> {
    outputs
        .iter()
        .enumerate()
        .map(|(idx, output)| {
            output
                .dimensions
                .iter()
                .enumerate()
                .map(|(jdx, dim)| match dim {
                    None => input_arrays[idx].shape()[jdx],
                    Some(d) => *d as usize,
                })
                .collect()
        })
        .collect()
}

fn arrays_to_tensors<T: Element, D: Dimension>(
    memory_info: &MemoryInfo,
    arrays: impl IntoIterator<Item = Array<T, D>>,
) -> Result<Vec<OrtTensor<T, D>>> {
    Ok(arrays
        .into_iter()
        .map(|arr| OrtTensor::from_array(memory_info, arr))
        .collect::<Result<Vec<_>>>()?)
}

fn tensors_to_ptr<'a, 's: 'a, T: Element, D: Dimension + 'a>(
    tensors: impl IntoIterator<Item = &'a OrtTensor<'s, T, D>>,
) -> Vec<*const sys::OrtValue> {
    tensors
        .into_iter()
        .map(|tensor| tensor.c_ptr as *const _)
        .collect()
}

fn tensors_to_mut_ptr<'a, 's: 'a, T: Element, D: Dimension + 'a>(
    tensors: impl IntoIterator<Item = &'a mut OrtTensor<'s, T, D>>,
) -> Vec<*mut sys::OrtValue> {
    tensors
        .into_iter()
        .map(|tensor| tensor.c_ptr as *mut _)
        .collect()
}

fn arrays_to_ort<T: Element, D: Dimension>(
    memory_info: &MemoryInfo,
    arrays: impl IntoIterator<Item = Array<T, D>>,
) -> Result<(Vec<OrtTensor<T, D>>, Vec<*const sys::OrtValue>)> {
    let ort_tensors = arrays
        .into_iter()
        .map(|arr| OrtTensor::from_array(memory_info, arr))
        .collect::<Result<Vec<_>>>()?;
    let ort_values = ort_tensors
        .iter()
        .map(|tensor| tensor.c_ptr as *const _)
        .collect();
    Ok((ort_tensors, ort_values))
}

fn arrays_with_shapes<T: Element, D: Dimension>(shapes: &[Vec<usize>]) -> Result<Vec<Array<T, D>>> {
    Ok(shapes
        .into_iter()
        .map(|shape| Array::<_, IxDyn>::default(shape.clone()).into_dimensionality())
        .collect::<std::result::Result<Vec<Array<T, D>>, _>>()?)
}

pub struct Inputs<'r, 'a, T: Element, D: Dimension> {
    tensors: &'a mut [OrtTensor<'r, T, D>],
}

impl<T: Element, D: Dimension> Inputs<'_, '_, T, D> {}

impl<T: Element, D: Dimension> Index<usize> for Inputs<'_, '_, T, D> {
    type Output = Array<T, D>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &(*self.tensors[index])
    }
}

impl<T: Element, D: Dimension> IndexMut<usize> for Inputs<'_, '_, T, D> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut (*self.tensors[index])
    }
}

pub struct Outputs<'r, 'a, T: Element, D: Dimension> {
    tensors: &'a [OrtTensor<'r, T, D>],
}

impl<T: Element, D: Dimension> Outputs<'_, '_, T, D> {}

impl<T: Element, D: Dimension> Index<usize> for Outputs<'_, '_, T, D> {
    type Output = Array<T, D>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &(*self.tensors[index])
    }
}

pub struct RunnerBuilder<'s, TIn: Element, DIn: Dimension> {
    session: &'s Session,
    input_arrays: Vec<Array<TIn, DIn>>,
}

impl<'s, TIn: Element, DIn: Dimension> RunnerBuilder<'s, TIn, DIn> {
    #[inline]
    pub fn new(
        session: &'s Session,
        input_arrays: impl IntoIterator<Item = Array<TIn, DIn>>,
    ) -> Self {
        Self {
            session,
            input_arrays: input_arrays.into_iter().collect(),
        }
    }

    #[inline]
    pub fn with_output<TOut: Element, DOut: Dimension>(
        self,
    ) -> Result<Runner<'s, TIn, DIn, TOut, DOut>> {
        Runner::new(self.session, self.input_arrays)
    }

    #[inline]
    pub fn with_output_dyn<TOut: Element>(self) -> Result<Runner<'s, TIn, DIn, TOut, IxDyn>> {
        Runner::new(self.session, self.input_arrays)
    }
}

pub struct Runner<'s, TIn: Element, DIn: Dimension, TOut: Element, DOut: Dimension> {
    session: &'s Session,
    input_names_ptr: Vec<*const i8>,
    output_names_ptr: Vec<*const i8>,
    input_ort_tensors: Vec<OrtTensor<'s, TIn, DIn>>,
    input_ort_values_ptr: Vec<*const sys::OrtValue>,
    output_ort_tensors: Vec<OrtTensor<'s, TOut, DOut>>,
    output_ort_values_ptr: Vec<*mut sys::OrtValue>,
}

impl<'s, TIn: Element, DIn: Dimension, TOut: Element, DOut: Dimension>
    Runner<'s, TIn, DIn, TOut, DOut>
{
    pub fn new(
        session: &'s Session,
        input_arrays: impl IntoIterator<Item = Array<TIn, DIn>>,
    ) -> Result<Self> {
        let input_names_ptr = names_to_ptrs(session.inputs.iter().map(|i| i.name.clone()));
        let output_names_ptr = names_to_ptrs(session.outputs.iter().map(|o| o.name.clone()));
        let input_arrays = input_arrays.into_iter().collect::<Vec<_>>();
        session.validate_input_shapes(&input_arrays)?;
        let output_shapes = compute_output_shapes(&input_arrays, &session.outputs);
        let output_arrays = arrays_with_shapes::<_, DOut>(&output_shapes)?;
        let input_ort_tensors = arrays_to_tensors(&session.memory_info, input_arrays)?;
        let input_ort_values_ptr = tensors_to_ptr(&input_ort_tensors);
        let mut output_ort_tensors = arrays_to_tensors(&session.memory_info, output_arrays)?;
        let output_ort_values_ptr = tensors_to_mut_ptr(&mut output_ort_tensors);
        Ok(Self {
            session,
            input_names_ptr,
            output_names_ptr,
            input_ort_tensors,
            input_ort_values_ptr,
            output_ort_tensors,
            output_ort_values_ptr,
        })
    }

    #[inline]
    pub fn inputs(&mut self) -> Inputs<'s, '_, TIn, DIn> {
        Inputs {
            tensors: self.input_ort_tensors.as_mut_slice(),
        }
    }

    #[inline]
    pub fn outputs(&'s self) -> Outputs<'s, '_, TOut, DOut> {
        Outputs {
            tensors: self.output_ort_tensors.as_slice(),
        }
    }

    #[inline]
    pub fn execute(&mut self) -> Result<()> {
        Ok(status_to_result(unsafe {
            g_ort().Run.unwrap()(
                self.session.session_ptr,
                std::ptr::null() as _,
                self.input_names_ptr.as_ptr(),
                self.input_ort_values_ptr.as_ptr(),
                self.input_ort_values_ptr.len() as _,
                self.output_names_ptr.as_ptr(),
                self.output_names_ptr.len() as _,
                self.output_ort_values_ptr.as_mut_ptr(),
            )
        })
        .map_err(OrtError::Run)?)
    }
}
