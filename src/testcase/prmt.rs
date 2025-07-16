use crate::test::{make_range, RangeTest, TestCase, TestCommon, TestPtx};

pub static PTX: &str = include_str!("prmt.ptx");

#[derive(Clone, Copy)]
pub enum PrmtMode {
    Generic,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

impl PrmtMode {
    fn as_str(self) -> &'static str {
        match self {
            PrmtMode::Generic => "",
            PrmtMode::F4e => ".f4e",
            PrmtMode::B4e => ".b4e",
            PrmtMode::Rc8 => ".rc8",
            PrmtMode::Ecl => ".ecl",
            PrmtMode::Ecr => ".ecr",
            PrmtMode::Rc16 => ".rc16",
        }
    }
}

pub fn all_tests() -> Vec<TestCase> {
    let mut tests = vec![];
    const MODES: [PrmtMode; 7] = [
        PrmtMode::Generic,
        PrmtMode::F4e,
        PrmtMode::B4e,
        PrmtMode::Rc8,
        PrmtMode::Ecl,
        PrmtMode::Ecr,
        PrmtMode::Rc16,
    ];
    for mode in MODES.iter() {
        let name = match mode {
            PrmtMode::Generic => "prmt".to_string(),
            _ => format!("prmt_{}", mode.as_str().trim_start_matches('.')),
        };
        tests.push(TestCase::new(name, make_range(Prmt { mode: *mode })));
    }
    tests
}

pub struct Prmt {
    pub mode: PrmtMode,
}

impl TestPtx for Prmt {
    fn body(&self) -> String {
        PTX.replace("<MODE>", self.mode.as_str())
    }

    fn args(&self) -> &[&str] {
        &["input0", "input1", "input2", "output"]
    }
}

impl TestCommon for Prmt {
    type Input = (u32, u32, u32);
    type Output = u32;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, c) = input;
        let expected = host_prmt(self.mode, a, b, c as u16);
        if expected == output {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

impl RangeTest for Prmt {
    const MAX_VALUE: u32 = u16::MAX as u32;

    fn generate(&self, input: u32) -> Self::Input {
        let c = input;
        let a = u32::from_ne_bytes([1, 2, 3, 4]);
        let b = u32::from_ne_bytes([5, 6, 7, 8]);
        (a, b, c)
    }
}

fn host_prmt(mode: PrmtMode, a: u32, b: u32, c: u16) -> u32 {
    let src = {
        let a_bytes = a.to_le_bytes();
        let b_bytes = b.to_le_bytes();
        [
            a_bytes[0], a_bytes[1], a_bytes[2], a_bytes[3], b_bytes[0], b_bytes[1], b_bytes[2],
            b_bytes[3],
        ]
    };
    match mode {
        PrmtMode::Generic => {
            let mut result = 0u32;
            for i in 0..4 {
                let nibble = ((c >> (i * 4)) & 0xF) as u8;
                let index = (nibble & 0x7) as usize;
                let sign = (nibble & 0x8) != 0;
                let byte = if sign {
                    // Sign-extend
                    if src[index] & 0x80 != 0 {
                        0xFF
                    } else {
                        0x00
                    }
                } else {
                    src[index]
                };
                result |= (byte as u32) << (i * 8);
            }
            result
        }
        PrmtMode::F4e => {
            let sel = (c & 0x3) as usize;
            let indices = match sel {
                0 => [3, 2, 1, 0],
                1 => [4, 3, 2, 1],
                2 => [5, 4, 3, 2],
                3 => [6, 5, 4, 3],
                _ => unreachable!(),
            };
            ((src[indices[0]] as u32) << 24)
                | ((src[indices[1]] as u32) << 16)
                | ((src[indices[2]] as u32) << 8)
                | (src[indices[3]] as u32)
        }
        PrmtMode::B4e => {
            let sel = (c & 0x3) as usize;
            let indices = match sel {
                0 => [5, 6, 7, 0],
                1 => [6, 7, 0, 1],
                2 => [7, 0, 1, 2],
                3 => [0, 1, 2, 3],
                _ => unreachable!(),
            };
            ((src[indices[0]] as u32) << 24)
                | ((src[indices[1]] as u32) << 16)
                | ((src[indices[2]] as u32) << 8)
                | (src[indices[3]] as u32)
        }
        PrmtMode::Rc8 => {
            let sel = (c & 0x3) as usize;
            let byte = src[sel];
            (byte as u32).wrapping_mul(0x01010101)
        }
        PrmtMode::Ecl => {
            let sel = (c & 0x3) as usize;
            let indices = match sel {
                0 => [3, 2, 1, 0],
                1 => [3, 2, 1, 1],
                2 => [3, 2, 2, 2],
                3 => [3, 3, 3, 3],
                _ => unreachable!(),
            };
            ((src[indices[0]] as u32) << 24)
                | ((src[indices[1]] as u32) << 16)
                | ((src[indices[2]] as u32) << 8)
                | (src[indices[3]] as u32)
        }
        PrmtMode::Ecr => {
            let sel = (c & 0x3) as usize;
            let indices = match sel {
                0 => [0, 0, 0, 0],
                1 => [1, 1, 1, 0],
                2 => [2, 2, 1, 0],
                3 => [3, 2, 1, 0],
                _ => unreachable!(),
            };
            ((src[indices[0]] as u32) << 24)
                | ((src[indices[1]] as u32) << 16)
                | ((src[indices[2]] as u32) << 8)
                | (src[indices[3]] as u32)
        }
        PrmtMode::Rc16 => {
            let sel = (c & 0x3) as usize;
            let indices = match sel {
                0 => [1, 0, 1, 0],
                1 => [3, 2, 3, 2],
                2 => [1, 0, 1, 0],
                3 => [3, 2, 3, 2],
                _ => unreachable!(),
            };
            ((src[indices[0]] as u32) << 24)
                | ((src[indices[1]] as u32) << 16)
                | ((src[indices[2]] as u32) << 8)
                | (src[indices[3]] as u32)
        }
    }
}
