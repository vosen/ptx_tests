use test::{run_random, Bfe};
mod bfe;
mod bfe2;
mod cuda;
mod test;

fn main() {
    let tests = [bfe::u32, bfe::s32, bfe::u64, bfe::s64];
    if run_random::<Bfe<u32>>() {
        println!("u32 PASS");
    } else {
        println!("u32 FAIL");
    }
}
