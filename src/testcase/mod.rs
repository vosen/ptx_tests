use crate::{cuda::Cuda, test::{TestCase, TestCommon}};

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

pub struct TestContext {
    pub cuda: Cuda,
}

impl TestContext {
    /// Turn argument names into PTX test function signature.
    fn fmt_ptx_signature(args: &[&str]) -> String {
        let args: Vec<_> = args.iter().map(|a| format!(".param .u64 {}", a)).collect();
        format!(".entry run({})", args.join(", "))
    }

    /// Prepare test sources to be loaded as a module on the device.
    pub fn prepare_test_source<T: TestCommon>(&self, t: &T) -> String {
        let body = t.ptx();
        let args = t.ptx_args();
        let header = t.ptx_header();

        format!(
            "{}\n{}\n{{\n{}\nret;\n}}\0",
            header,
            Self::fmt_ptx_signature(args),
            body,
        )
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
    tests
}
