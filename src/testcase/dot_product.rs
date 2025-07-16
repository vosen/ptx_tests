use crate::test::{make_random, RandomTest, TestCase, TestCommon, TestPtx};
use rand::Rng;

pub static PTX: &str = include_str!("dot_product.ptx");

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];

    // dp4a tests
    for &signed in &[false, true] {
        let name = format!("dp4a_{}", if signed { "s32" } else { "u32" });
        tests.push(TestCase::new(
            name,
            make_random(DotProd {
                dp2a: false,
                signed,
                hi: false,
            }),
        ));
    }

    // dp2a tests
    for &signed in &[false, true] {
        for &hi in &[false, true] {
            let name = format!(
                "dp2a_{}_{}",
                if hi { "hi" } else { "lo" },
                if signed { "s32" } else { "u32" },
            );
            tests.push(TestCase::new(
                name,
                make_random(DotProd {
                    dp2a: true,
                    signed,
                    hi,
                }),
            ));
        }
    }
    tests
}

#[derive(Default, Clone, Copy)]
pub struct DotProd {
    pub dp2a: bool, // false = dp4a
    pub signed: bool,
    pub hi: bool, // only used when dp2a is true: true is hi, false is lo
}

impl TestPtx for DotProd {
    fn body(&self) -> String {
        let typ = if self.signed { "s32" } else { "u32" };
        let instr = if !self.dp2a {
            format!("dp4a.{}.{}", typ, typ)
        } else {
            let mode = if self.hi { "hi" } else { "lo" };
            format!("dp2a.{}.{}.{}", mode, typ, typ)
        };
        PTX.replace("<DPINST>", &instr).replace("<TYPE>", typ)
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
            dp4a(a, b, c_raw, self.signed)
        } else {
            dp2a(a, b, c_raw, self.hi, self.signed)
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

fn dp4a(a: u32, b: u32, c: u32, signed: bool) -> u32 {
    let a_bytes = a.to_le_bytes();
    let b_bytes = b.to_le_bytes();
    if !signed {
        let mut acc = c;
        for i in 0..4 {
            let va = a_bytes[i] as u32;
            let vb = b_bytes[i] as u32;
            acc = acc.wrapping_add(va.wrapping_mul(vb));
        }
        acc
    } else {
        let mut acc = c as i32;
        for i in 0..4 {
            let va = (a_bytes[i] as i8) as i32;
            let vb = (b_bytes[i] as i8) as i32;
            acc = acc.wrapping_add(va.wrapping_mul(vb));
        }
        acc as u32
    }
}

fn dp2a(a: u32, b: u32, c: u32, hi: bool, signed: bool) -> u32 {
    let a_bytes = a.to_le_bytes();
    let b_bytes = b.to_le_bytes();
    if !signed {
        let a_lo = u16::from_le_bytes([a_bytes[0], a_bytes[1]]) as u32;
        let a_hi = u16::from_le_bytes([a_bytes[2], a_bytes[3]]) as u32;
        let (b0, b1) = if hi {
            (b_bytes[2] as u32, b_bytes[3] as u32)
        } else {
            (b_bytes[0] as u32, b_bytes[1] as u32)
        };
        c.wrapping_add(a_lo.wrapping_mul(b0).wrapping_add(a_hi.wrapping_mul(b1)))
    } else {
        let a_lo = i16::from_le_bytes([a_bytes[0], a_bytes[1]]) as i32;
        let a_hi = i16::from_le_bytes([a_bytes[2], a_bytes[3]]) as i32;
        let (b0, b1) = if hi {
            ((b_bytes[2] as i8) as i32, (b_bytes[3] as i8) as i32)
        } else {
            ((b_bytes[0] as i8) as i32, (b_bytes[1] as i8) as i32)
        };
        let acc = (c as i32)
            .wrapping_add(a_lo.wrapping_mul(b0))
            .wrapping_add(a_hi.wrapping_mul(b1));
        acc as u32
    }
}
