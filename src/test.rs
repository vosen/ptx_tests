use half::f16;
use num::{Bounded, Num, PrimInt, Zero};
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::{any::Any, fmt::Debug, mem, ptr, u32};

use crate::{cuda::CUmodule, TestContext};

pub trait TestPtx {
    fn args(&self) -> &[&str];
    fn body(&self) -> String;
}

pub trait TestCommon: TestPtx {
    type Input: OnDevice + DebugRich;
    type Output: OnDevice + DebugRich;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output>;
}

pub trait RangeTest: TestCommon {
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(&self, input: u32) -> Self::Input;
}

pub trait RandomTest: TestCommon + Default {
    fn generate<R: Rng>(rng: &mut R) -> Self::Input;
}

pub trait OnDevice: Copy + Debug {
    const COMPONENTS: usize;
    fn write(self, buffers: &mut [Vec<u8>]);
    fn read(buffers: &[Vec<u8>], index: usize) -> Self;
    fn size_of() -> usize {
        mem::size_of::<Self>()
    }
    fn zero() -> Self {
        unsafe { mem::zeroed::<Self>() }
    }
}

impl OnDevice for u8 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for u16 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for i16 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}

impl OnDevice for u32 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for i32 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for u64 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for i64 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for f16 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for f32 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl OnDevice for f64 {
    const COMPONENTS: usize = 1;

    fn write(self, buffers: &mut [Vec<u8>]) {
        buffers[0].extend_from_slice(&self.to_le_bytes());
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        unsafe {
            buffers[0]
                .as_ptr()
                .cast::<Self>()
                .add(index)
                .read_unaligned()
        }
    }
}
impl<X: OnDevice, Y: OnDevice> OnDevice for (X, Y) {
    const COMPONENTS: usize = 2;

    fn write(self, buffers: &mut [Vec<u8>]) {
        self.0.write(&mut buffers[0..]);
        self.1.write(&mut buffers[1..]);
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        (X::read(&buffers[0..], index), Y::read(&buffers[1..], index))
    }

    fn size_of() -> usize {
        X::size_of() + Y::size_of()
    }

    fn zero() -> Self {
        (X::zero(), Y::zero())
    }
}

impl<X: OnDevice, Y: OnDevice, Z: OnDevice> OnDevice for (X, Y, Z) {
    const COMPONENTS: usize = 3;

    fn write(self, buffers: &mut [Vec<u8>]) {
        self.0.write(&mut buffers[0..]);
        self.1.write(&mut buffers[1..]);
        self.2.write(&mut buffers[2..]);
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        (
            X::read(&buffers[0..], index),
            Y::read(&buffers[1..], index),
            Z::read(&buffers[2..], index),
        )
    }

    fn size_of() -> usize {
        X::size_of() + Y::size_of() + Z::size_of()
    }

    fn zero() -> Self {
        (X::zero(), Y::zero(), Z::zero())
    }
}

impl<X: OnDevice, Y: OnDevice, Z: OnDevice, W: OnDevice> OnDevice for (X, Y, Z, W) {
    const COMPONENTS: usize = 4;

    fn write(self, buffers: &mut [Vec<u8>]) {
        self.0.write(&mut buffers[0..]);
        self.1.write(&mut buffers[1..]);
        self.2.write(&mut buffers[2..]);
        self.3.write(&mut buffers[3..]);
    }

    fn read(buffers: &[Vec<u8>], index: usize) -> Self {
        (
            X::read(&buffers[0..], index),
            Y::read(&buffers[1..], index),
            Z::read(&buffers[2..], index),
            W::read(&buffers[3..], index),
        )
    }

    fn size_of() -> usize {
        X::size_of() + Y::size_of() + Z::size_of() + W::size_of()
    }

    fn zero() -> Self {
        (X::zero(), Y::zero(), Z::zero(), W::zero())
    }
}

pub trait DebugRich {
    fn debug_rich(&self) -> String;
}

macro_rules! impl_debug_rich {
    ($type:ident) => {
        impl DebugRich for $type {
            fn debug_rich(&self) -> String {
                format!("{self:#066b} {self:#X} {self}")
            }
        }
    }
}

impl_debug_rich!(u16);
impl_debug_rich!(i16);
impl_debug_rich!(u32);
impl_debug_rich!(i32);
impl_debug_rich!(u64);
impl_debug_rich!(i64);

impl DebugRich for f16 {
    fn debug_rich(&self) -> String {
        format!("{self:#066b} {self:#X} {self:.24}")
    }
}

impl DebugRich for f32 {
    fn debug_rich(&self) -> String {
        let bits = self.to_bits();
        format!("{bits:#066b} {bits:#X} {self:.24}")
    }
}

impl DebugRich for f64 {
    fn debug_rich(&self) -> String {
        let bits = self.to_bits();
        format!("{bits:#066b} {bits:#X} {self:.24}")
    }
}

impl<T: DebugRich> DebugRich for (T,) {
    fn debug_rich(&self) -> String {
        self.0.debug_rich()
    }
}

impl<T1, T2> DebugRich for (T1, T2)
where
    T1: DebugRich,
    T2: DebugRich,
{
    fn debug_rich(&self) -> String {
        format!(
            "(\n{},\n{},\n)",
            self.0.debug_rich(),
            self.1.debug_rich(),
        )
    }
}

impl<T1, T2, T3> DebugRich for (T1, T2, T3)
where
    T1: DebugRich,
    T2: DebugRich,
    T3: DebugRich,
{
    fn debug_rich(&self) -> String {
        format!(
            "(\n{},\n{},\n{},\n)",
            self.0.debug_rich(),
            self.1.debug_rich(),
            self.2.debug_rich(),
        )
    }
}

impl<T1, T2, T3, T4> DebugRich for (T1, T2, T3, T4)
where
    T1: DebugRich,
    T2: DebugRich,
    T3: DebugRich,
    T4: DebugRich,
{
    fn debug_rich(&self) -> String {
        format!(
            "(\n{},\n{},\n{},\n{},\n)",
            self.0.debug_rich(),
            self.1.debug_rich(),
            self.2.debug_rich(),
            self.3.debug_rich(),
        )
    }
}

pub trait PtxScalar: Copy + Num + Bounded + Debug + DebugRich + OnDevice + Any {
    fn name() -> &'static str;

    fn unsigned() -> bool {
        Self::min_value() == <Self as Zero>::zero()
    }
    fn float() -> bool {
        false
    }
    fn signed() -> bool {
        !Self::float() && !Self::unsigned()
    }
    fn is_f32() -> bool {
        Self::float() && Self::size_of() == 4
    }
}

impl PtxScalar for u16 {
    fn name() -> &'static str {
        "u16"
    }
}

impl PtxScalar for i16 {
    fn name() -> &'static str {
        "s16"
    }
}

impl PtxScalar for u32 {
    fn name() -> &'static str {
        "u32"
    }
}

impl PtxScalar for i32 {
    fn name() -> &'static str {
        "s32"
    }
}

impl PtxScalar for u64 {
    fn name() -> &'static str {
        "u64"
    }
}

impl PtxScalar for i64 {
    fn name() -> &'static str {
        "s64"
    }
}

impl PtxScalar for f16 {
    fn name() -> &'static str {
        "f16"
    }
    fn float() -> bool {
        true
    }
}

impl PtxScalar for f32 {
    fn name() -> &'static str {
        "f32"
    }
    fn float() -> bool {
        true
    }
}

impl PtxScalar for f64 {
    fn name() -> &'static str {
        "f64"
    }
    fn float() -> bool {
        true
    }
}

const SEED: u64 = 0x761194f3027874ef;
const GROUP_SIZE: usize = 128;
// Totally unscientific number that works on my machine
const SAFE_MEMORY_LIMIT: usize = 1 << 29;

fn load_module(ctx: &dyn TestContext, t: &dyn TestPtx) -> Result<CUmodule, TestError> {
    let cuda = ctx.cuda();

    match ctx.prepare_test_source(t) {
        Ok(src) => {
            let mut module = ptr::null_mut();
            let load_result = unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) };

            match load_result {
                Ok(()) => Ok(module),
                Err(code) => return Err(TestError::CompilationFail { message: format!("CUDA Error {code}") }),
            }
        },
        Err(message) => return Err(TestError::CompilationFail { message }),
    }
}

pub fn run_random<T: RandomTest>(ctx: &dyn TestContext) -> Result<(), TestError> {
    let cuda = ctx.cuda();
    let t =  T::default();

    let module = load_module(ctx, &t)?;
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module, c"run".as_ptr()) }.unwrap();

    let mut rng = XorShiftRng::seed_from_u64(SEED);
    let mut free_memory = 0;
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut free_memory, &mut total_memory) }.unwrap();
    let max_memory = (total_memory / 2).min(SAFE_MEMORY_LIMIT);
    let total_elements = 2.pow(32);
    assert!(total_elements % GROUP_SIZE == 0);
    let element_size = T::Input::size_of() + T::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size: usize =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    let mut inputs = vec![Vec::new(); T::Input::COMPONENTS];
    let mut outputs = vec![T::Output::zero(); memory_batch_size / element_size];
    for iteration in 0..iterations {
        assert_eq!(T::Output::COMPONENTS, 1);
        let memory_batch_size = if iteration == iterations - 1 {
            required_memory - (memory_batch_size * (iterations - 1))
        } else {
            memory_batch_size
        };
        let element_batch_size = memory_batch_size / element_size;
        for vec in inputs.iter_mut() {
            vec.clear();
        }
        for _ in 0..element_batch_size {
            T::generate(&mut rng).write(&mut inputs);
        }
        outputs.resize(element_batch_size, T::Output::zero());
        let dev_inputs: Vec<u64> = inputs
            .iter()
            .map(|vec| {
                let mut devptr = 0;
                unsafe { cuda.cuMemAlloc_v2(&mut devptr, vec.len()) }.unwrap();
                unsafe { cuda.cuMemcpyHtoD_v2(devptr, vec.as_ptr().cast_mut().cast(), vec.len()) }
                    .unwrap();
                devptr
            })
            .collect();
        let mut dev_output = 0;
        unsafe { cuda.cuMemAlloc_v2(&mut dev_output, element_batch_size * T::Output::size_of()) }
            .unwrap();
        let mut args = dev_inputs
            .iter()
            .map(|ptr| ptr as *const u64)
            .collect::<Vec<_>>();
        args.push(&dev_output);
        unsafe {
            cuda.cuLaunchKernel(
                kernel,
                (element_batch_size / GROUP_SIZE) as u32,
                1,
                1,
                GROUP_SIZE as u32,
                1,
                1,
                0,
                0 as _,
                args.as_mut_ptr() as _,
                ptr::null_mut(),
            )
        }
        .unwrap();
        unsafe { cuda.cuStreamSynchronize(0 as _) }.unwrap();
        unsafe {
            cuda.cuMemcpyDtoH_v2(
                outputs.as_mut_ptr() as _,
                dev_output,
                outputs.len() * T::Output::size_of(),
            )
        }
        .unwrap();
        for (i, output) in outputs.iter().copied().enumerate() {
            let input = T::Input::read(&inputs, i);
            if let Err(expected) = t.host_verify(input, output) {
                return Err(TestError::ResultMismatch {
                    input: input.debug_rich(),
                    output: output.debug_rich(),
                    expected: expected.debug_rich(),
                });
            }
        }
        for devptr in dev_inputs {
            unsafe { cuda.cuMemFree_v2(devptr) }.unwrap();
        }
        unsafe { cuda.cuMemFree_v2(dev_output) }.unwrap();
    }
    unsafe { cuda.cuModuleUnload(module) }.unwrap();

    Ok(())
}

fn next_multiple_of(value: usize, multiple: usize) -> usize {
    ((value + multiple - 1) / multiple) * multiple
}

pub fn run_range<Test: RangeTest>(ctx: &dyn TestContext, t: Test) -> Result<(), TestError> {
    let cuda = ctx.cuda();

    let module = load_module(ctx, &t)?;
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module, c"run".as_ptr()) }.unwrap();

    let mut free_memory = 0;
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut free_memory, &mut total_memory) }.unwrap();
    let max_memory = (total_memory / 2).min(SAFE_MEMORY_LIMIT);
    let total_elements = Test::MAX_VALUE as usize + 1;
    if total_elements % GROUP_SIZE != 0 {
        panic!(
            "Total element count {} must be divisble by {}",
            total_elements, GROUP_SIZE
        );
    }
    let element_size = Test::Input::size_of() + Test::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size: usize =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    let mut inputs = vec![Vec::new(); Test::Input::COMPONENTS];
    let mut outputs = vec![Test::Output::zero(); memory_batch_size / element_size];
    for iteration in 0..iterations {
        assert_eq!(Test::Output::COMPONENTS, 1);
        let elment_start = iteration * memory_batch_size / element_size;
        let memory_batch_size = if iteration == iterations - 1 {
            required_memory - (memory_batch_size * (iterations - 1))
        } else {
            memory_batch_size
        };
        let element_batch_size = memory_batch_size / element_size;
        for vec in inputs.iter_mut() {
            vec.clear();
        }
        for i in 0..element_batch_size {
            let input = t.generate((elment_start + i) as u32);
            input.write(&mut inputs);
        }
        outputs.resize(element_batch_size, Test::Output::zero());
        let dev_inputs: Vec<u64> = inputs
            .iter()
            .map(|vec| {
                let mut devptr = 0;
                unsafe { cuda.cuMemAlloc_v2(&mut devptr, vec.len()) }.unwrap();
                unsafe { cuda.cuMemcpyHtoD_v2(devptr, vec.as_ptr().cast_mut().cast(), vec.len()) }
                    .unwrap();
                devptr
            })
            .collect();
        let mut dev_output = 0;
        unsafe {
            cuda.cuMemAlloc_v2(
                &mut dev_output,
                element_batch_size * Test::Output::size_of(),
            )
        }
        .unwrap();
        let mut args = dev_inputs
            .iter()
            .map(|ptr| ptr as *const u64)
            .collect::<Vec<_>>();
        args.push(&dev_output);
        unsafe {
            cuda.cuLaunchKernel(
                kernel,
                (element_batch_size / GROUP_SIZE) as u32,
                1,
                1,
                GROUP_SIZE as u32,
                1,
                1,
                0,
                0 as _,
                args.as_mut_ptr() as _,
                ptr::null_mut(),
            )
        }
        .unwrap();
        unsafe { cuda.cuStreamSynchronize(0 as _) }.unwrap();
        unsafe {
            cuda.cuMemcpyDtoH_v2(
                outputs.as_mut_ptr() as _,
                dev_output,
                outputs.len() * Test::Output::size_of(),
            )
        }
        .unwrap();
        for (i, output) in outputs.iter().copied().enumerate() {
            let input = Test::Input::read(&inputs, i);
            if let Err(expected) = t.host_verify(input, output) {
                return Err(TestError::ResultMismatch {
                    input: input.debug_rich(),
                    output: output.debug_rich(),
                    expected: expected.debug_rich(),
                });
            }
        }
        for devptr in dev_inputs {
            unsafe { cuda.cuMemFree_v2(devptr) }.unwrap();
        }
        unsafe { cuda.cuMemFree_v2(dev_output) }.unwrap();
    }
    unsafe { cuda.cuModuleUnload(module) }.unwrap();

    Ok(())
}

pub type TestFunction = Box<dyn FnOnce(&dyn TestContext) -> Result<(), TestError>>;

pub fn make_random<T: RandomTest>() -> TestFunction {
    return Box::new(|ctx| run_random::<T>(ctx));
}

pub fn make_range<T: RangeTest + 'static>(t: T) -> TestFunction {
    return Box::new(move |ctx| run_range::<T>(ctx, t));
}

pub struct TestCase {
    pub test: TestFunction,
    pub name: String,
}

impl TestCase {
    pub fn new(name: String, test: TestFunction) -> Self {
        TestCase { test, name }
    }

    pub fn join_invalid_tests(
        name: String,
        tests: Vec<(
            String,
            TestFunction,
        )>,
    ) -> Self {
        use TestError::*;

        let test = Box::new(move |ctx: &dyn TestContext| {
            for (name, test) in tests {
                match test(ctx) {
                    Err(CompilationFail { .. }) => {},
                    Ok(()) | Err(ResultMismatch { .. }) => return Err(CompilationSuccess { name }),
                    Err(CompilationSuccess { .. }) => unreachable!("tests may not report CompilationSuccess"),
                }
            }
            Ok(())
        });
        TestCase { test, name }
    }
}

/// Errors that a test can produce.
pub enum TestError {
    /// Used when compilation fails, e.g. during CUDA module loading or NVRTC launch
    CompilationFail {
        message: String,
    },
    /// Used when tests that should have failed compilation, succeed unexpectedly
    CompilationSuccess {
        name: String,
    },
    /// Used when the test compiled successfully, but found mismatching values
    ResultMismatch {
        input: String,
        output: String,
        expected: String,
    },
}
