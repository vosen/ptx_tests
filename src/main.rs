mod bfe;
mod cuda;

fn main() {
    let tests = [bfe::u32, bfe::s32, bfe::u64, bfe::s64];
    if bfe::u32() {
        println!("u32 PASS");
    } else {
        println!("u32 FAIL");
    }
}
