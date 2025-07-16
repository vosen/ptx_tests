use crate::test::{RangeTest, TestCase, TestCommon, TestPtx};

static PTX: &str = include_str!("sad.ptx");

#[derive(Clone, Copy)]
struct Sad {
    pub signed: bool,
}

impl TestPtx for Sad {
    fn body(&self) -> String {
        let typ = if self.signed { "s16" } else { "u16" };
        PTX.replace("<TYPE>", typ)
    }

    fn args(&self) -> &[&str] {
        &["input0", "input1", "input2", "output"]
    }
}

impl TestCommon for Sad {
    type Input = (u16, u16, u16);
    type Output = u16;

    fn host_verify(&self, input: Self::Input, output: Self::Output) -> Result<(), Self::Output> {
        let (a, b, c) = input;
        if !self.signed {
            let diff = if a < b {
                b.wrapping_sub(a)
            } else {
                a.wrapping_sub(b)
            };
            let expected = c.wrapping_add(diff);
            if expected == output {
                Ok(())
            } else {
                Err(expected)
            }
        } else {
            let a_signed = a as i16;
            let b_signed = b as i16;
            let c_signed = c as i16;
            let diff = if a_signed < b_signed {
                b_signed.wrapping_sub(a_signed)
            } else {
                a_signed.wrapping_sub(b_signed)
            };
            let expected = c_signed.wrapping_add(diff);
            let expected_u16 = expected as u16;
            if expected_u16 == output {
                Ok(())
            } else {
                Err(expected_u16)
            }
        }
    }
}

impl RangeTest for Sad {
    const MAX_VALUE: u32 = u32::MAX;

    fn generate(&self, index: u32) -> Self::Input {
        // Lower 16 bits for c.
        let c = (index & 0xFFFF) as u16;

        // Next 8 bits for a (top 4 and bottom 4), upper 8 bits for b.
        let a_byte = ((index >> 16) & 0xFF) as u8;
        let b_byte = ((index >> 24) & 0xFF) as u8;

        let a_top = a_byte >> 4;
        let a_bottom = a_byte & 0x0F;
        let a = ((a_top as u16) << 12) | (a_bottom as u16);

        let b_top = b_byte >> 4;
        let b_bottom = b_byte & 0x0F;
        let b = ((b_top as u16) << 12) | (b_bottom as u16);

        (a, b, c)
    }
}

pub fn all_tests() -> Vec<TestCase> {
    vec![
        TestCase::new(
            "sad_u16".to_string(),
            crate::test::make_range(Sad { signed: false }),
        ),
        TestCase::new(
            "sad_s16".to_string(),
            crate::test::make_range(Sad { signed: true }),
        ),
    ]
}
