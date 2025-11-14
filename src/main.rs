#![allow(internal_features)]
#![feature(link_llvm_intrinsics)]
#![feature(f16)]
#![feature(c_size_t)]

use std::ptr;

use bpaf::Bpaf;
use nvrtc::Nvrtc;
use regex::{self, Regex};

use cuda::Cuda;
use test::{TestCase, TestError};
use testcase::*;

mod common;
mod cuda;
mod nvrtc;
mod test;
mod testcase;

#[derive(Debug, Clone, Bpaf)]
#[bpaf(options)]
enum Arguments {
    List {
        /// list all available tests, execute no tests
        #[bpaf(short, long)]
        #[allow(dead_code)]
        list: (),
    },
    Run {
        /// only tests matching this regex will be executed
        #[bpaf(short, long)]
        filter: Option<String>,

        /// path to NVRTC shared library, switches to testing inline PTX embedded in CUDA sources when provided
        #[bpaf(long)]
        nvrtc: Option<String>,

        /// number of shards to split the tests into for parallel execution
        #[bpaf(external, optional)]
        shards: Option<Shards>,

        /// fail on the first test failure
        fail_fast: bool,

        /// path to CUDA shared library under testing, for example C:\Windows\System32\nvcuda.dll or /usr/lib/x86_64-linux-gnu/libcuda.so
        #[bpaf(positional("cuda"))]
        cuda: String,
    },
}

#[derive(Debug, Clone, Bpaf)]
struct Shards {
    /// index of the shard to run, starting from 0
    shard_index: usize,
    /// total number of shards to split the tests into for parallel execution
    shard_count: usize,
}

fn main() {
    let args = arguments().run();

    let mut tests = tests();

    match args {
        Arguments::List { .. } => {
            for test in tests {
                println!("{}", test.name);
            }
        }
        Arguments::Run {
            filter,
            nvrtc,
            cuda,
            shards,
            fail_fast,
        } => {
            if let Some(filter) = filter {
                let re = Regex::new(&filter).unwrap();
                tests = tests.into_iter().filter(|t| re.is_match(&t.name)).collect();
            }
            let tests = if let Some(shards) = shards {
                let start = shards.shard_index * tests.len() / shards.shard_count;
                let end = (shards.shard_index + 1) * tests.len() / shards.shard_count;
                tests.drain(start..end).collect()
            } else {
                tests
            };

            let cuda = Cuda::new(cuda);
            let nvrtc = nvrtc.map(Nvrtc::new);

            let failures = if let Some(nvrtc) = nvrtc {
                let libs = (cuda, nvrtc);
                run(tests, TestFixture { libs }, fail_fast)
            } else {
                let libs = (cuda,);
                run(tests, TestFixture { libs }, fail_fast)
            };

            std::process::exit(failures);
        }
    }
}

fn run(tests: Vec<TestCase>, ctx: impl TestContext, fail_fast: bool) -> i32 {
    let cuda = ctx.cuda();

    let mut failures = 0;

    unsafe { cuda.cuInit(0) }.unwrap();
    let mut cuda_ctx = ptr::null_mut();
    unsafe { cuda.cuCtxCreate_v2(&mut cuda_ctx, 0, 0) }.unwrap();

    for t in tests {
        use TestError::*;

        let result = (t.test)(&ctx, fail_fast);
        if result.is_err() {
            failures += 1;
        }

        print!("{}: ", t.name);
        match result {
            Ok(()) => println!("OK"),
            Err(CompilationFail { message }) => println!("FAIL - Compilation failed:\n{message}"),
            Err(CompilationSuccess { name }) => {
                println!("FAIL - Compilation mismatch, didn't expect '{name}' to compile")
            }
            Err(ResultMismatch {
                input,
                output,
                expected,
                total_cases,
                passed_cases,
            }) => {
                let percent = (passed_cases as f32 / total_cases as f32) * 100f32;
                println!(
                    "FAIL - with input {input}\n    computed on GPU: {output}\n    computed on CPU: {expected}\n    passed: {passed_cases} out of {total_cases} ({percent}%)"
                )
            }
            Err(MissingRunFunction) => println!("FAIL - Missing run function"),
        }
    }

    failures
}

#[macro_export]
macro_rules! impl_library {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* $(,)* ) -> $ret_type:ty);* $(;)*) => {
        $(
            #[allow(non_snake_case)]
            #[allow(improper_ctypes)]
            pub unsafe fn $fn_name(&self,  $( $arg_id : $arg_type),*) -> $ret_type {
                let fn_: libloading::Symbol<unsafe extern $abi fn( $($arg_type),*) -> $ret_type> =
                    self.library.get(concat!(stringify!($fn_name), "\0").as_bytes()).unwrap();
                fn_( $($arg_id),*)
            }
        )*
    };
}
