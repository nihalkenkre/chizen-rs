fn main() {
    let from = std::env::var("CARGO_MANIFEST_DIR").unwrap().to_owned();
    let mut to = from.clone() + "\\target";
    println!("cargo:warning=copy {} -> {}", from.clone(), to);

    let mut cmd = "python ".to_owned() + &from.clone() + "\\build_slang.py " + &from.clone();

    if cfg!(debug_assertions) {
        to += "\\debug";
        cmd += " Debug "
    } else {
        to += "\\release";
        cmd += " Release "
    }

    cmd += &(to.clone());

    println!("cargo:warning={}", cmd.clone());
    std::process::Command::new("cmd")
        .args(&["/C", &cmd])
        .output()
        .unwrap();
}
