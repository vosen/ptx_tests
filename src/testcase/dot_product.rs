use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};
use rand::Rng;

pub static PTX: &str = include_str!("dot_product.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];

    // dp4a tests
    for &a_signed in &[false, true] {
        for &b_signed in &[false, true] {
            let atype = if a_signed { "s32" } else { "u32" };
            let btype = if b_signed { "s32" } else { "u32" };
            let name = format!("dp4a_{atype}_{btype}");
            tests.push(TestCase::new(
                name,
                make_random(DotProd {
                    dp2a: false,
                    a_signed,
                    b_signed,
                    hi: false,
                }),
            ));
        }
    }

    // dp2a tests
    for &a_signed in &[false, true] {
        for &b_signed in &[false, true] {
            for &hi in &[false, true] {
                let mode = if hi { "hi" } else { "lo" };
                let atype = if a_signed { "s32" } else { "u32" };
                let btype = if b_signed { "s32" } else { "u32" };
                let name = format!("dp2a_{mode}_{atype}_{btype}",);
                tests.push(TestCase::new(
                    name,
                    make_random(DotProd {
                        dp2a: true,
                        a_signed,
                        b_signed,
                        hi,
                    }),
                ));
            }
        }
    }
    tests
}

#[derive(Default, Clone, Copy)]
pub struct DotProd {
    pub dp2a: bool, // false = dp4a
    pub a_signed: bool,
    pub b_signed: bool,
    pub hi: bool, // only used when dp2a is true: true is hi, false is lo
}

impl TestPtx for DotProd {
    fn body(&self) -> String {
        let atype = if self.a_signed { "s32" } else { "u32" };
        let btype = if self.b_signed { "s32" } else { "u32" };
        let instr = if !self.dp2a {
            format!("dp4a.{}.{}", atype, btype)
        } else {
            let mode = if self.hi { "hi" } else { "lo" };
            format!("dp2a.{}.{}.{}", mode, atype, btype)
        };
        PTX.replace("<DPINST>", &instr)
    }

    fn args(&self) -> &[&str] {
        &["input0", "input1", "input2", "output"]
    }
}

impl TestCommon for DotProd {
    type Input = (u32, u32, u32); //generate as unsigned, just reinterpret as signed in dp4a/dp2a functions (and CUDA)
    type Output = u32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, c_raw) = input;
        let expected = if !self.dp2a {
            dp4a(a, b, c_raw, self.a_signed, self.b_signed)
        } else {
            dp2a(a, b, c_raw, self.hi, self.a_signed, self.b_signed)
        };
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RandomTest for DotProd {
    fn generate<R: Rng>(&self, rng: &mut R) -> Self::Input {
        (rng.next_u32(), rng.next_u32(), rng.next_u32())
    }
}

fn dp4a(a: u32, b: u32, c: u32, a_signed: bool, b_signed: bool) -> u32 {
    let a = sext_or_zext_u8(a, a_signed);
    let b = sext_or_zext_u8(b, b_signed);
    a.into_iter()
        .zip(b.into_iter())
        .fold(c, |acc, (a, b)| acc.wrapping_add(a.wrapping_mul(b)))
}

fn sext_or_zext_u8(x: u32, signed: bool) -> [u32; 4] {
    let bytes = x.to_le_bytes();
    if signed {
        // sign-extend
        [
            bytes[0] as i8 as u32,
            bytes[1] as i8 as u32,
            bytes[2] as i8 as u32,
            bytes[3] as i8 as u32,
        ]
    } else {
        // zero-extend
        [
            bytes[0] as u32,
            bytes[1] as u32,
            bytes[2] as u32,
            bytes[3] as u32,
        ]
    }
}

fn sext_or_zext_u16(x: u32, signed: bool) -> [u32; 2] {
    let bytes = x.to_le_bytes();
    if signed {
        // sign-extend
        [
            i16::from_le_bytes([bytes[0], bytes[1]]) as i32 as u32,
            i16::from_le_bytes([bytes[2], bytes[3]]) as i32 as u32,
        ]
    } else {
        // zero-extend
        [
            u16::from_le_bytes([bytes[0], bytes[1]]) as u32,
            u16::from_le_bytes([bytes[2], bytes[3]]) as u32,
        ]
    }
}

fn dp2a(a: u32, b: u32, c: u32, hi: bool, a_signed: bool, b_signed: bool) -> u32 {
    let a = sext_or_zext_u16(a, a_signed);
    let b = sext_or_zext_u8(b, b_signed);
    let b = if hi { [b[2], b[3]] } else { [b[0], b[1]] };
    a.into_iter()
        .zip(b.into_iter())
        .fold(c, |acc, (a, b)| acc.wrapping_add(a.wrapping_mul(b)))
}
