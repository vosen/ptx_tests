use std::ffi::CString;

use crate::{cuda::Cuda, test::{TestCase, TestPtx}};

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
    fn prepare_test_source(&self, ptx: &dyn TestPtx) -> Result<CString, String>;
}

pub struct TestFixture<L> {
    pub libs: L,
}

const PTX_HEADER: &'_ str = "
    .version 7.0
    .target sm_80
    .address_size 64
";

impl TestContext for TestFixture<(Cuda,)> {
    fn cuda(&self) -> &Cuda {
        &self.libs.0
    }

    fn prepare_test_source(&self, ptx: &dyn TestPtx) -> Result<CString, String> {
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

        Ok(CString::new(format!(
            "{}\n{}\n{{\n{}\n{}\nret;\n}}",
            PTX_HEADER,
            fmt_ptx_signature(ptx.args()),
            fmt_ptx_params_load(ptx.args()),
            ptx.body(),
        )).unwrap())
    }
}

pub fn tests() -> Vec<TestCase> {
    let mut tests = vec![];
    tests.extend(bfe::all_tests());
    tests.extend(bfi::all_tests());
    tests.extend(brev::all_tests());
    tests.extend(cvt::all_tests());
    tests.extend(rcp::all_tests());
    tests.extend(shift::all_tests());
    tests.extend(minmax::all_tests());
    tests.extend(sqrt::all_tests());
    tests.extend(rsqrt::all_tests());
    tests.extend(sin::all_tests());
    tests.extend(cos::all_tests());
    tests.extend(lg2::all_tests());
    tests
}
