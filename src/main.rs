#![allow(internal_features)]
#![feature(link_llvm_intrinsics)]
#![feature(f16)]

use bpaf::Bpaf;
use cuda::Cuda;
use regex::{self, Regex};
use std::ptr;
use test::{TestCase, TestError};

mod bfe;
mod bfi;
mod brev;
mod common;
mod cos;
mod cuda;
mod cvt;
mod lg2;
mod minmax;
mod rcp;
mod rsqrt;
mod shift;
mod sin;
mod sqrt;
mod test;

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options)]
pub enum Arguments {
    List {
        /// list all available tests, execute no tests
        #[bpaf(short, long)]
        list: (),
    },
    Run {
        /// only tests matching this regex will be executed
        #[bpaf(short, long)]
        filter: Option<String>,

        /// path to CUDA shared library under testing, for example C:\Windows\System32\nvcuda.dll or /usr/lib/x86_64-linux-gnu/libcuda.so
        #[bpaf(positional("cuda"))]
        cuda: String,
    },
}

fn main() {
    let args = arguments().run();
    std::process::exit(run(args));
}

fn tests() -> Vec<TestCase> {
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

fn run(args: Arguments) -> i32 {
    let mut failures = 0;
    let mut tests = tests();
    tests.sort_unstable_by_key(|t| t.name.clone());
    match args {
        Arguments::List { .. } => {
            for test in tests {
                println!("{}", test.name);
            }
        }
        Arguments::Run { filter, cuda } => {
            if let Some(filter) = filter {
                let re = Regex::new(&filter).unwrap();
                tests = tests.into_iter().filter(|t| re.is_match(&t.name)).collect();
            }
            let cuda = Cuda::new(cuda);
            unsafe { cuda.cuInit(0) }.unwrap();
            let mut ctx = ptr::null_mut();
            unsafe { cuda.cuCtxCreate_v2(&mut ctx, 0, 0) }.unwrap();
            for t in tests {
                match (t.test)(&cuda) {
                    Ok(()) => println!("{}: OK", t.name),
                    Err(TestError::Mismatch(e)) => {
                        println!(
                            "{}: FAIL: Input {}, computed on GPU {}, computed on CPU {}",
                            t.name, e.input, e.output, e.expected
                        );
                        failures += 1;
                    }
                    Err(TestError::Miscompile(name)) => {
                        println!("{}: FAIL: Compilation mismatch", name);
                        failures += 1;
                    }
                }
            }
        }
    }
    failures
}
