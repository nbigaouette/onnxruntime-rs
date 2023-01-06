use crate::{
    environment::Environment,
    error::{assert_not_null_pointer, assert_null_pointer, status_to_result},
    g_ort, OrtError, Result,
};
use onnxruntime_sys as sys;
use tracing::{debug, error};

#[derive(Debug)]
pub struct ModelMetadata {
    pub(crate) _env: Environment,
    pub(crate) allocator_ptr: *mut sys::OrtAllocator,
    pub(crate) metadata_ptr: *mut sys::OrtModelMetadata,
}

unsafe impl Send for ModelMetadata {}

impl ModelMetadata {
    fn _get_string(
        &self,
        f: impl Fn(
            *const sys::OrtModelMetadata,
            *mut sys::OrtAllocator,
            *mut *mut std::ffi::c_char,
        ) -> *mut sys::OrtStatus,
    ) -> Result<String> {
        let mut output_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
        let status = unsafe { f(self.metadata_ptr, self.allocator_ptr, &mut output_ptr) };

        status_to_result(status).map_err(OrtError::ModelMetadata)?;
        assert_null_pointer(status, "MetadataStatus")?;
        assert_not_null_pointer(self.metadata_ptr, "Metadata")?;

        let str = unsafe { std::ffi::CStr::from_ptr(output_ptr) }
            .to_string_lossy()
            .to_string();
        unsafe {
            g_ort().AllocatorFree.unwrap()(self.allocator_ptr, output_ptr as *mut std::ffi::c_void)
        };
        Ok(str)
    }

    pub fn producer_name(&self) -> Result<String> {
        self._get_string(|metadata, allocator, buf| unsafe {
            g_ort().ModelMetadataGetProducerName.unwrap()(metadata, allocator, buf)
        })
    }

    pub fn graph_name(&self) -> Result<String> {
        self._get_string(|metadata, allocator, buf| unsafe {
            g_ort().ModelMetadataGetGraphName.unwrap()(metadata, allocator, buf)
        })
    }

    pub fn domain(&self) -> Result<String> {
        self._get_string(|metadata, allocator, buf| unsafe {
            g_ort().ModelMetadataGetDomain.unwrap()(metadata, allocator, buf)
        })
    }

    pub fn description(&self) -> Result<String> {
        self._get_string(|metadata, allocator, buf| unsafe {
            g_ort().ModelMetadataGetDescription.unwrap()(metadata, allocator, buf)
        })
    }

    pub fn version(&self) -> Result<i64> {
        let mut version: i64 = 0;
        let status =
            unsafe { g_ort().ModelMetadataGetVersion.unwrap()(self.metadata_ptr, &mut version) };

        status_to_result(status).map_err(OrtError::ModelMetadata)?;
        assert_null_pointer(status, "MetadataStatus")?;
        assert_not_null_pointer(self.metadata_ptr, "Metadata")?;

        Ok(version)
    }

    pub fn custom_metadata(&self, key: &str) -> Result<String> {
        let c_key = std::ffi::CString::new(key)?;
        self._get_string(|metadata, allocator, buf| unsafe {
            g_ort().ModelMetadataLookupCustomMetadataMap.unwrap()(
                metadata,
                allocator,
                c_key.as_ptr() as *const std::ffi::c_char,
                buf,
            )
        })
    }
}

impl Drop for ModelMetadata {
    #[tracing::instrument]
    fn drop(&mut self) {
        if self.metadata_ptr.is_null() {
            error!("Model metadata is null, not dropping");
        } else {
            debug!("Dropping the model metadata.");
            unsafe { g_ort().ReleaseModelMetadata.unwrap()(self.metadata_ptr) };
        }
    }
}
