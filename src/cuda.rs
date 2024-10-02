use libloading::{Library, Symbol};
use std::num::NonZeroU32;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUctx_ {
    _unused: [u8; 0],
}
pub type CUcontext = *mut CUctx_;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmod_ {
    _unused: [u8; 0],
}
pub type CUmodule = *mut CUmod_;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUfunc_ {
    _unused: [u8; 0],
}
pub type CUfunction = *mut CUfunc_;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_ {
    _unused: [u8; 0],
}
pub type CUstream = *mut CUstream_;

pub type CUdeviceptr = ::std::os::raw::c_ulonglong;

pub type CUdevice = ::std::os::raw::c_int;

pub type CUresult = Result<(), NonZeroU32>;
static_assertions::assert_eq_size!(CUresult, u32);

pub struct Cuda {
    library: Library,
}

impl Cuda {
    pub fn new(path: String) -> Self {
        let library = unsafe { libloading::Library::new(path).unwrap() };
        Self { library }
    }
}

macro_rules! cuda_impl {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* $(,)* ) -> $ret_type:path);* $(;)*) => {
        impl Cuda {
            $(
                #[allow(non_snake_case)]
                #[allow(improper_ctypes)]
                pub unsafe fn $fn_name(&self,  $( $arg_id : $arg_type),*) -> $ret_type {
                    let fn_: Symbol<unsafe extern $abi fn( $($arg_type),*) -> $ret_type> =
                        self.library.get(concat!(stringify!($fn_name), "\0").as_bytes()).unwrap();
                    fn_( $($arg_id),*)
                }
            )*
        }
    };
}

cuda_impl! {
    "system" fn cuInit(Flags: ::std::os::raw::c_uint) -> CUresult;
    "system" fn cuCtxCreate_v2(
        pctx: *mut CUcontext,
        flags: ::std::os::raw::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    "system" fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const ::std::os::raw::c_void,
    ) -> CUresult;
    "system" fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const ::std::os::raw::c_char,
    ) -> CUresult;
    "system" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    "system" fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::std::os::raw::c_void,
        ByteCount: usize,
    ) -> CUresult;
    //"system" fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: ::std::os::raw::c_uchar, N: usize)
    //    -> CUresult;
    "system" fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::std::os::raw::c_uint,
        gridDimY: ::std::os::raw::c_uint,
        gridDimZ: ::std::os::raw::c_uint,
        blockDimX: ::std::os::raw::c_uint,
        blockDimY: ::std::os::raw::c_uint,
        blockDimZ: ::std::os::raw::c_uint,
        sharedMemBytes: ::std::os::raw::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::std::os::raw::c_void,
        extra: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;
    "system" fn cuMemcpyDtoH_v2(
        dstHost: *mut ::std::os::raw::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    "system" fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    "system" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    "system" fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    //"system" fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    "system" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
}
