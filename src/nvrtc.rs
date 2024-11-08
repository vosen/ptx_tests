#![allow(non_camel_case_types)]

use std::num::NonZeroU32;

use libloading::Library;

use crate::impl_library;


pub type nvrtcResult = Result<(), NonZeroU32>;
static_assertions::assert_eq_size!(nvrtcResult, u32);


#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvrtcProgram_ {
    _unused: [u8; 0],
}
pub type nvrtcProgram = *mut nvrtcProgram_;


pub struct Nvrtc {
    library: Library,
}

impl Nvrtc {
    pub fn new(path: String) -> Self {
        let library = unsafe { Library::new(path).unwrap() };
        Self { library }
    }
}

impl Nvrtc {
    impl_library! {
        "system" fn nvrtcCreateProgram(
            prog: *mut nvrtcProgram,
            src: *const ::core::ffi::c_char,
            name: *const ::core::ffi::c_char,
            numHeaders: ::core::ffi::c_int,
            headers: *const *const ::core::ffi::c_char,
            includeNames: *const *const ::core::ffi::c_char,
        ) -> nvrtcResult;
        "system" fn nvrtcCompileProgram(
            prog: nvrtcProgram,
            numOptions: ::core::ffi::c_int,
            options: *const *const ::core::ffi::c_char,
        ) -> nvrtcResult;
        "system" fn nvrtcDestroyProgram(
            prog: *mut nvrtcProgram,
        ) -> nvrtcResult;
        "system" fn nvrtcGetPTXSize(
            prog: nvrtcProgram,
            ptxSizeRet: *mut ::core::ffi::c_size_t,
        ) -> nvrtcResult;
        "system" fn nvrtcGetPTX(
            prog: nvrtcProgram,
            ptx: *mut ::core::ffi::c_char,
        ) -> nvrtcResult;
    }
}
