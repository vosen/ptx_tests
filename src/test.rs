use num::{cast::AsPrimitive, traits::FromBytes, Bounded, Num, PrimInt, Zero};
use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::{fmt::Debug, mem, ptr, u32};

use crate::cuda::Cuda;

trait TestCommon {
    type Input: OnDevice;
    type Output: OnDevice;
    fn host_verify(input: Self::Input, output: Self::Output) -> bool;
    fn ptx() -> String;
}

trait RangeTest: TestCommon {
    const MAX_VALUE: u32 = u32::MAX;
    fn generate(input: u32) -> Self::Input;
}

trait RandomTest: TestCommon {
    fn generate<R: Rng>(rng: &mut R) -> Self::Input;
}

trait OnDevice: Copy + Debug {
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

pub struct Bfe<T: PtxScalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: PtxScalar> TestCommon for Bfe<T> {
    type Input = (T, u32, u32);

    type Output = T;

    fn ptx() -> String {
        let mut src = crate::bfe2::PTX
            .replace("<TYPE>", T::name())
            .replace("<TYPE_SIZE>", &mem::size_of::<T>().to_string());
        src.push('\0');
        src
    }

    fn host_verify(input: Self::Input, output: Self::Output) -> bool {
        fn bfe_host<T: PtxScalar>(value: T, pos: u32, len: u32) -> T {
            let pos = pos.to_le_bytes()[0] as usize;
            let len = len.to_le_bytes()[0] as usize;
            let msb = mem::size_of::<T>() * 8 - 1;
            let sbit = if T::unsigned() || len == 0 {
                false
            } else {
                get_bit(value, Ord::min(pos + len - 1, pos))
            };
            let mut d = <T as Zero>::zero();
            for i in 0..=msb {
                let bit = if i < len && pos + i <= msb {
                    get_bit(value, pos + i)
                } else {
                    sbit
                };
                set_bit(&mut d, i, bit)
            }
            d
        }
        let (value, len, pos) = input;
        bfe_host(value, len, pos) == output
    }
}

impl<T: PtxScalar + FromBytes> RangeTest for Bfe<T>
where
    for<'a> T::Bytes: TryFrom<&'a [u8]>,
    for<'a> <<T as FromBytes>::Bytes as TryFrom<&'a [u8]>>::Error: Debug,
{
    fn generate(input: u32) -> Self::Input {
        let len = input.to_le_bytes()[0] as u32;
        let pos = input.to_le_bytes()[1] as u32;
        let value = [
            input.to_le_bytes()[3],
            input.to_le_bytes()[2],
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let value = T::from_be_bytes(&T::Bytes::try_from(&value).unwrap());
        (value, pos, len)
    }
}

impl<T: PtxScalar> RandomTest for Bfe<T>
where
    Standard: Distribution<T>,
{
    fn generate<R: Rng + ?Sized>(rng: &mut R) -> Self::Input {
        let value = rng.gen();
        let len = rng.gen::<u32>();
        let pos = rng.gen::<u32>();
        (value, len, pos)
    }
}

pub trait PtxScalar:
    Copy + Num + Bounded + PrimInt + AsPrimitive<usize> + Debug + OnDevice
{
    fn name() -> &'static str;
    fn unsigned() -> bool {
        Self::min_value() == <Self as Zero>::zero()
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

fn get_bit<T: PtxScalar>(value: T, n: usize) -> bool {
    assert!(n < mem::size_of::<T>() * 8);
    let value: usize = value.as_();
    value & (1 << n) != 0
}

fn set_bit<T: PtxScalar>(value: &mut T, n: usize, bit: bool) {
    assert!(n < mem::size_of::<T>() * 8);
    let mask = T::one().unsigned_shl(n as u32);
    if bit {
        *value = value.bitor(mask);
    } else {
        *value = value.bitand(mask.not());
    }
}

const SEED: u64 = 0x761194f3027874ef;
const GROUP_SIZE: usize = 128;

pub fn run_random<T: RandomTest>() -> bool {
    let mut src = T::ptx();
    let cuda = Cuda::new();
    unsafe { cuda.cuInit(0) }.unwrap();
    let mut ctx = ptr::null_mut();
    unsafe { cuda.cuCtxCreate_v2(&mut ctx, 0, 0) }.unwrap();
    let mut module = ptr::null_mut();
    unsafe { cuda.cuModuleLoadData(&mut module, src.as_ptr() as _) }.unwrap();
    let mut kernel = ptr::null_mut();
    unsafe { cuda.cuModuleGetFunction(&mut kernel, module, c"bfe".as_ptr()) }.unwrap();
    let mut rng = XorShiftRng::seed_from_u64(SEED);
    let mut total_memory = 0;
    unsafe { cuda.cuMemGetInfo_v2(&mut 0, &mut total_memory) }.unwrap();
    let max_memory = total_memory / 2;
    let total_elements = u32::MAX as usize + 1;
    assert!(total_elements % GROUP_SIZE == 0);
    let element_size = T::Input::size_of() + T::Output::size_of();
    let required_memory = total_elements * element_size;
    let iterations = (required_memory / max_memory).max(1);
    let memory_batch_size =
        next_multiple_of(required_memory / iterations, GROUP_SIZE * element_size);
    for iteration in 0..iterations {
        let mut inputs = vec![Vec::new(); T::Input::COMPONENTS];
        assert_eq!(T::Output::COMPONENTS, 1);
        let memory_batch_size = if iteration == iterations - 1 {
            required_memory - (memory_batch_size * (iterations - 1))
        } else {
            memory_batch_size
        };
        let element_batch_size = memory_batch_size / element_size;
        for _ in 0..element_batch_size {
            T::generate(&mut rng).write(&mut inputs);
        }
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
        let mut result = vec![T::Output::zero(); element_batch_size];
        unsafe {
            cuda.cuMemcpyDtoH_v2(
                result.as_mut_ptr() as _,
                dev_output,
                result.len() * T::Output::size_of(),
            )
        }
        .unwrap();
        for (i, result) in result.iter().copied().enumerate() {
            let value = T::Input::read(&inputs, i);
            let result = result;
            if !T::host_verify(value, result) {
                panic! {"Mismatch for value {:?} and result {:?}", value, result};
            }
        }
        for devptr in dev_inputs {
            unsafe { cuda.cuMemFree_v2(devptr) }.unwrap();
        }
        unsafe { cuda.cuMemFree_v2(dev_output) }.unwrap();
    }
    true
}

fn next_multiple_of(value: usize, multiple: usize) -> usize {
    ((value + multiple - 1) / multiple) * multiple
}
