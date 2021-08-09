#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"
#if !defined(__APPLE__)
#include "cuda_provider_factory.h"
#endif