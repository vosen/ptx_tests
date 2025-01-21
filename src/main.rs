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

        /// path to NVRTC shared library, switches to testing inline PTX embedded in CUDA sources when provided
        #[bpaf(long)]
        nvrtc: Option<String>,

        /// path to CUDA shared library under testing, for example C:\Windows\System32\nvcuda.dll or /usr/lib/x86_64-linux-gnu/libcuda.so
        #[bpaf(positional("cuda"))]
        cuda: String,
    },
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
        Arguments::Run { filter, nvrtc, cuda } => {
            if let Some(filter) = filter {
                let re = Regex::new(&filter).unwrap();
                tests = tests.into_iter().filter(|t| re.is_match(&t.name)).collect();
            }

            let cuda = Cuda::new(cuda);
            let nvrtc = nvrtc.map(Nvrtc::new);

            let failures = if let Some(nvrtc) = nvrtc {
                let libs = (cuda, nvrtc);
                run(tests, TestFixture { libs })
            } else {
                let libs = (cuda,);
                run(tests, TestFixture { libs })
            };

            std::process::exit(failures);
        }
    }
}

fn run(tests: Vec<TestCase>, ctx: impl TestContext) -> i32 {
    let cuda = ctx.cuda();

    let mut failures = 0;

    unsafe { cuda.cuInit(0) }.unwrap();
    let mut cuda_ctx = ptr::null_mut();
    unsafe { cuda.cuCtxCreate_v2(&mut cuda_ctx, 0, 0) }.unwrap();

    for t in tests {
        use TestError::*;

        let result = (t.test)(&ctx);
        if result.is_err() {
            failures += 1;
        }

        print!("{}: ", t.name);
        match result {
            Ok(()) => println!("OK"),
            Err(CompilationFail { message }) => println!("FAIL - Compilation failed:\n{message}"),
            Err(CompilationSuccess { name }) => println!("FAIL - Compilation mismatch, didn't expect '{name}' to compile"),
            Err(ResultMismatch { input, output, expected }) => println!(
                "FAIL - with input {input}\n    computed on GPU: {output}\n    computed on CPU: {expected}"
            ),
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
