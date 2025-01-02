use std::{alloc::{alloc, dealloc, Layout}, ffi::{CStr, CString}, ptr};

use crate::{cuda::Cuda, nvrtc::Nvrtc, test::{TestCase, TestPtx}};

mod bfe;
mod bfi;
mod brev;
mod cos;
mod cvt;
mod lg2;
mod minmax;
mod rcp;
mod rsqrt;
mod shift;
mod sin;
mod sqrt;

pub trait TestContext {
    fn cuda(&self) -> &Cuda;
    fn prepare_test_source(&self, ptx: &dyn TestPtx) -> CString;
}

pub struct TestFixture<L> {
    pub libs: L,
}

impl TestContext for TestFixture<(Cuda,)> {
    fn cuda(&self) -> &Cuda {
        &self.libs.0
    }

    fn prepare_test_source(&self, ptx: &dyn TestPtx) -> CString {
        /// Generate PTX test function signature.
        fn fmt_ptx_signature(args: &[&str]) -> String {
            let args: Vec<_> = args.iter().map(|a| format!(".param .u64 {}", a)).collect();
            format!(".entry run({})", args.join(", "))
        }

        /// Generate PTX to load values of test function parameters.
        fn fmt_ptx_params_load(args: &[&str]) -> String {
            let mut text = String::new();
            for arg in args {
                text.push_str(&format!(".reg .u64    {name}_addr;\n", name = arg));
                text.push_str(&format!("ld.param.u64 {name}_addr, [{name}];\n", name = arg));
            }
            text
        }

        let source = format!(
            "{}\n{}\n{{\n{}\n{}\nret;\n}}",
            ptx.header(),
            fmt_ptx_signature(ptx.args()),
            fmt_ptx_params_load(ptx.args()),
            ptx.body(),
        );

        CString::new(source).unwrap()
    }
}

impl TestContext for TestFixture<(Cuda, Nvrtc)> {
    fn cuda(&self) -> &Cuda {
        &self.libs.0
    }

    fn prepare_test_source(&self, ptx: &dyn TestPtx) -> CString {
        /// Generate CUDA test function signature.
        fn fmt_cuda_signature(args: &[&str]) -> String {
            let args: Vec<_> = args.iter().map(|a| format!("unsigned long long * {}", a)).collect();
            format!("extern \"C\" __global__ void run({})", args.join(", "))
        }

        /// Generate PTX to load values of test function parameters.
        fn fmt_cuda_inline_ptx_params_load(args: &[&str]) -> String {
            let mut text = String::new();
            for (arg_index, arg_name) in args.iter().enumerate() {
                text.push_str(&format!(".reg .u64 {name}_addr;\n", name = arg_name));
                text.push_str(&format!("mov.u64   {name}_addr, %{index};\n", name = arg_name, index = arg_index));
            }
            text
        }

        /// Generate CUDA parameter list for inline PTX.
        fn fmt_cuda_inline_ptx_params(args: &[&str]) -> String {
            args.iter().map(|a| format!(r#""l"({})"#, a)).collect::<Vec<_>>().join(", ")
        }

        /// Transform raw PTX into CUDA inline PTX function body.
        fn ptx_to_inline(args: &[&str], body: &str) -> String {
            let mut body = body.to_string();

            // Escape "%" (used for things like %tid (thread id) etc.)
            body = body.replace("%", "%%");

            body = format!(
                "{}\n{}",
                fmt_cuda_inline_ptx_params_load(args),
                body,
            );

            body = body.lines().map(|l| format!("\"{}\"\n", l)).collect::<Vec<_>>().join("    ");

            format!(
                "asm({}    :: {});",
                body,
                fmt_cuda_inline_ptx_params(args),
            )
        }

        let nvrtc = &self.libs.1;

        let source_cuda = format!(
            "{} {{\n{}\n}}",
            fmt_cuda_signature(ptx.args()),
            ptx_to_inline(ptx.args(), &ptx.body()),
        );
        let source_cuda = CString::new(source_cuda).unwrap();

        let mut program = ptr::null_mut();
        unsafe { nvrtc.nvrtcCreateProgram(&mut program, source_cuda.as_ptr() as _, ptr::null() as _, 0, ptr::null(), ptr::null()) }.unwrap();
        let result = unsafe { nvrtc.nvrtcCompileProgram(program, 0, ptr::null()) };

        if result.is_err() {
            let error = unsafe { CStr::from_ptr(nvrtc.nvrtcGetErrorString(result)) };
            let error = String::from_utf8_lossy(error.to_bytes()).to_string();

            eprintln!("NVRTC error: {error}");

            let mut log_size = 0;
            unsafe { nvrtc.nvrtcGetProgramLogSize(program, &mut log_size) }.unwrap();

            let log_layout = Layout::array::<core::ffi::c_char>(log_size).unwrap();
            let log_buffer = unsafe { alloc(log_layout) };

            unsafe { nvrtc.nvrtcGetProgramLog(program, log_buffer as _) }.unwrap();

            let log_cstr = unsafe { CStr::from_ptr(log_buffer as _) };
            let log = String::from_utf8_lossy(log_cstr.to_bytes()).to_string();

            eprintln!("Compilation produced the following log:\n{log}");

            unsafe { dealloc(log_buffer, log_layout) };

            std::process::exit(1);
        }

        let mut ptx_size = 0;
        unsafe { nvrtc.nvrtcGetPTXSize(program, &mut ptx_size) }.unwrap();

        let source_ptx_layout = Layout::array::<core::ffi::c_char>(ptx_size).unwrap();
        let source_ptx_buffer = unsafe { alloc(source_ptx_layout) };

        unsafe { nvrtc.nvrtcGetPTX(program, source_ptx_buffer as _ ) }.unwrap();

        let source_ptx = unsafe { CStr::from_ptr(source_ptx_buffer as _) };
        let source_ptx = source_ptx.to_owned();

        unsafe { nvrtc.nvrtcDestroyProgram(&mut program) }.unwrap();

        unsafe { dealloc(source_ptx_buffer, source_ptx_layout) };

        source_ptx
    }
}

pub fn tests() -> Vec<TestCase> {
    let mut tests = vec![
        bfe::rng_u32(),
        bfe::rng_s32(),
        bfe::rng_u64(),
        bfe::rng_s64(),
        bfi::rng_b32(),
        bfi::rng_b64(),
        brev::b32(),
    ];
    tests.extend(cvt::all_tests());
    tests.extend(rcp::all_tests());
    tests.extend(shift::all_tests());
    tests.extend(minmax::all_tests());
    tests.extend(sqrt::all_tests());
    tests.extend(rsqrt::all_tests());
    tests.extend(sin::all_tests());
    tests.extend(cos::all_tests());
    tests.extend(lg2::all_tests());

    tests.sort_unstable_by_key(|t| t.name.clone());

    tests
}
