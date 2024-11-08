use crate::test::TestCase;

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
